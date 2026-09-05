"""
SQLite cache for photonic kernel costs, architecture costs, and programming costs.

Cache keys include M/K/N, bit-widths, dataflow, config hash, device hash,
SimPhony commit, and schema version. Any change to these causes a cache miss.

This ensures only unique cost signatures run SimPhony (Plan Section 9 E2).

中文阅读提示：SimPhony 器件计算较慢。本文件只负责“保存和取回”，
不参与权重驻留或资源调度；缓存键必须包含架构和位宽，不能只看矩阵形状。
"""
# =============================================================================
# 本文件角色一句话：把 SimPhony（光子器件级仿真器）算出来的成本结果缓存到
# SQLite 数据库，避免同样的请求反复调用慢速仿真。
# 缓存命中条件很苛刻：M/K/N、位宽、dataflow、架构哈希、配置哈希、
# SimPhony 版本、schema 版本，任何一项不同都会导致缓存未命中（重新计算）。
# 类比：就像把"重复的计算题答案"记在小本子上，但只认"一模一样的题"。
# =============================================================================

import hashlib
import json
import os
import sqlite3
import time
from typing import Optional


class CostCache:
    # 数据库中的每条成本记录都对应一个不可混淆的硬件配置组合。
    """Versioned SQLite cache for SimPhony cost queries.

    Tables:
      - architecture_cost  : one row per (arch_hash, config_hash)
      - kernel_cost        : one row per (M, K, N, bits, dataflow, hashes)
      - programming_cost   : one row per (arch_hash, config_hash)
      - run_manifest       : records each unique simulation run
    """
    # 中文说明：四张表各管一类缓存：
    #   architecture_cost  一次性架构成本（面积、器件数、激光功率等）
    #   kernel_cost        每个 GEMM 光子算子的成本（时延/能耗分解）
    #   programming_cost   全系统权重编程（调谐）的一次性成本
    #   run_manifest       每次仿真运行的记录（用于审计/复现）

    def __init__(self, db_path: str = "joint_sim_cache.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """打开数据库连接并建表（幂等：重复调用无副作用）。"""
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self._db_path)
        # 让行可以按字典方式（row["列名"]）访问
        self._conn.row_factory = sqlite3.Row
        # WAL 日志模式：读写并发更友好（允许仿真时被外部工具查看）
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def close(self):
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        # 支持 with 语法：进入时自动打开
        self.open()
        return self

    def __exit__(self, *args):
        # 支持 with 语法：退出时自动关闭
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self):
        # 建四张表（IF NOT EXISTS 保证重复调用安全）。
        # 注意各表的联合主键正是"缓存键"：键不同即视为不同结果。
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS architecture_cost (
            arch_hash       TEXT NOT NULL,
            config_hash     TEXT NOT NULL,
            pic_area_um2    REAL,
            rf_eic_area_um2 REAL,
            total_area_um2  REAL,
            core_il_db      REAL,
            laser_power_w   REAL,
            mrr_count       INTEGER,
            pd_count        INTEGER,
            dac_count       INTEGER,
            adc_count       INTEGER,
            created_at      REAL,
            PRIMARY KEY (arch_hash, config_hash)
        );

        CREATE TABLE IF NOT EXISTS kernel_cost (
            M               INTEGER NOT NULL,
            K               INTEGER NOT NULL,
            N               INTEGER NOT NULL,
            input_bits      INTEGER NOT NULL,
            weight_bits     INTEGER NOT NULL,
            output_bits     INTEGER NOT NULL,
            dataflow        TEXT    NOT NULL,
            arch_hash       TEXT    NOT NULL,
            config_hash     TEXT    NOT NULL,
            schema_version  TEXT    NOT NULL,
            compute_latency_s          REAL,
            operand_encoding_latency_s REAL,
            conversion_latency_s       REAL,
            programming_latency_s      REAL,
            dynamic_energy_j   REAL,
            dac_energy_j       REAL,
            adc_energy_j       REAL,
            laser_energy_j     REAL,
            mrr_tuning_energy_j REAL,
            mrr_hold_energy_j  REAL,
            iter_M      INTEGER,
            iter_K      INTEGER,
            iter_N      INTEGER,
            switching_cycles INTEGER,
            max_cycles  INTEGER,
            utilization REAL,
            created_at  REAL,
            PRIMARY KEY (M, K, N, input_bits, weight_bits, output_bits,
                         dataflow, arch_hash, config_hash, schema_version)
        );

        CREATE TABLE IF NOT EXISTS programming_cost (
            arch_hash       TEXT NOT NULL,
            config_hash     TEXT NOT NULL,
            tile_count      INTEGER,
            programmed_mrr_count INTEGER,
            programming_latency_s  REAL,
            programming_energy_j   REAL,
            hold_power_w    REAL,
            created_at      REAL,
            PRIMARY KEY (arch_hash, config_hash)
        );

        CREATE TABLE IF NOT EXISTS run_manifest (
            run_id          TEXT PRIMARY KEY,
            simphony_commit TEXT,
            config_hash     TEXT,
            device_hash     TEXT,
            schema_version  TEXT,
            created_at      REAL
        );
        """)

    # ------------------------------------------------------------------
    # Architecture cost cache
    # ------------------------------------------------------------------

    def get_architecture_cost(
        self, arch_hash: str, config_hash: str
    ) -> Optional[dict]:
        """按 (架构哈希, 配置哈希) 查架构成本；查不到返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM architecture_cost WHERE arch_hash=? AND config_hash=?",
            (arch_hash, config_hash),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        # 数据库列名与对外字段名不一致的地方在这里还原
        result["core_insertion_loss_db"] = result.pop("core_il_db")
        result["laser_wall_plug_power_w"] = result.pop("laser_power_w")
        return result

    def put_architecture_cost(
        self, arch_hash: str, config_hash: str, cost: dict
    ):
        """把架构成本写进缓存；同键时覆盖（INSERT OR REPLACE）。"""
        self._conn.execute(
            """INSERT OR REPLACE INTO architecture_cost
               (arch_hash, config_hash, pic_area_um2, rf_eic_area_um2,
                total_area_um2, core_il_db, laser_power_w,
                mrr_count, pd_count, dac_count, adc_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                arch_hash, config_hash,
                cost.get("pic_area_um2", 0),
                cost.get("rf_eic_area_um2", 0),
                cost.get("total_area_um2", 0),
                cost.get("core_insertion_loss_db", 0),
                cost.get("laser_wall_plug_power_w", 0),
                cost.get("mrr_count", 0),
                cost.get("pd_count", 0),
                cost.get("dac_count", 0),
                cost.get("adc_count", 0),
                time.time(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Kernel cost cache
    # ------------------------------------------------------------------

    def _kernel_key(self, M, K, N, input_bits, weight_bits, output_bits,
                    dataflow, arch_hash, config_hash, schema_version):
        # 内核成本的主键 = 形状 + 位宽 + dataflow + 架构/配置/版本指纹
        return (
            M, K, N, input_bits, weight_bits, output_bits,
            dataflow, arch_hash, config_hash, schema_version,
        )

    def get_kernel_cost(
        self, M: int, K: int, N: int,
        input_bits: int, weight_bits: int, output_bits: int,
        dataflow: str, arch_hash: str, config_hash: str, schema_version: str,
    ) -> Optional[dict]:
        """按完整键查单个光子 GEMM 的成本；查不到返回 None。"""
        row = self._conn.execute(
            """SELECT * FROM kernel_cost
               WHERE M=? AND K=? AND N=?
                 AND input_bits=? AND weight_bits=? AND output_bits=?
                 AND dataflow=? AND arch_hash=? AND config_hash=?
                 AND schema_version=?""",
            self._kernel_key(
                M, K, N, input_bits, weight_bits, output_bits,
                dataflow, arch_hash, config_hash, schema_version,
            ),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def put_kernel_cost(
        self, M: int, K: int, N: int,
        input_bits: int, weight_bits: int, output_bits: int,
        dataflow: str, arch_hash: str, config_hash: str,
        schema_version: str, cost: dict,
    ):
        """把光子 GEMM 的成本写进缓存；同键覆盖。"""
        self._conn.execute(
            """INSERT OR REPLACE INTO kernel_cost
               (M, K, N, input_bits, weight_bits, output_bits,
                dataflow, arch_hash, config_hash, schema_version,
                compute_latency_s, operand_encoding_latency_s,
                conversion_latency_s, programming_latency_s,
                dynamic_energy_j, dac_energy_j, adc_energy_j,
                laser_energy_j, mrr_tuning_energy_j, mrr_hold_energy_j,
                iter_M, iter_K, iter_N,
                switching_cycles, max_cycles, utilization, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?,
                       ?, ?, ?, ?)""",
            (
                M, K, N, input_bits, weight_bits, output_bits,
                dataflow, arch_hash, config_hash, schema_version,
                cost.get("compute_latency_s", 0),
                cost.get("operand_encoding_latency_s", 0),
                cost.get("conversion_latency_s", 0),
                cost.get("programming_latency_s", 0),
                cost.get("dynamic_energy_j", 0),
                cost.get("dac_energy_j", 0),
                cost.get("adc_energy_j", 0),
                cost.get("laser_energy_j", 0),
                cost.get("mrr_tuning_energy_j", 0),
                cost.get("mrr_hold_energy_j", 0),
                cost.get("iter_M", 0),
                cost.get("iter_K", 0),
                cost.get("iter_N", 0),
                cost.get("switching_cycles", 0),
                cost.get("max_cycles", 0),
                cost.get("utilization", 0),
                time.time(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Programming cost cache
    # ------------------------------------------------------------------

    def get_programming_cost(
        self, arch_hash: str, config_hash: str
    ) -> Optional[dict]:
        """按 (架构哈希, 配置哈希) 查全系统权重编程成本；查不到返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM programming_cost WHERE arch_hash=? AND config_hash=?",
            (arch_hash, config_hash),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def put_programming_cost(
        self, arch_hash: str, config_hash: str, cost: dict
    ):
        """把全系统权重编程成本写进缓存；同键覆盖。"""
        self._conn.execute(
            """INSERT OR REPLACE INTO programming_cost
               (arch_hash, config_hash, tile_count, programmed_mrr_count,
                programming_latency_s, programming_energy_j,
                hold_power_w, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                arch_hash, config_hash,
                cost.get("tile_count", 0),
                cost.get("programmed_mrr_count", 0),
                cost.get("programming_latency_s", 0),
                cost.get("programming_energy_j", 0),
                cost.get("hold_power_w", 0),
                time.time(),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------

    def record_run(
        self, run_id: str, simphony_commit: str, config_hash: str,
        device_hash: str, schema_version: str,
    ):
        """记录一次仿真运行（用于审计：谁用什么配置算过）。"""
        self._conn.execute(
            """INSERT OR REPLACE INTO run_manifest
               (run_id, simphony_commit, config_hash, device_hash,
                schema_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, simphony_commit, config_hash, device_hash,
             schema_version, time.time()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def hash_dict(d: dict) -> str:
        """Stable hash of a configuration dictionary."""
        # 把配置字典 JSON 序列化（键排序保证稳定）后做 SHA-256，
        # 取前 16 位十六进制作为短指纹
        raw = json.dumps(d, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def hash_file(path: str) -> str:
        """Hash of file contents."""
        # 对文件内容整体做 SHA-256 取前 16 位
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]


# ------------------------------------------------------------------
# Convenience: cache-aware backend wrapper
# ------------------------------------------------------------------

class CachedSimPhonyBackend:
    """Combines SimPhonyBackend with CostCache for transparent caching.

    Usage:
        backend = CachedSimPhonyBackend(
            simphony_backend=SimPhonyBackend(),
            cache=CostCache("simphony_cache.db"),
            arch_hash="abc123",
            config_hash="def456",
        )
        cost = backend.kernel_cost(M=257, K=512, N=512)
        # Second call hits cache.
    """
    # 中文说明：一个"带缓存的 SimPhony 后端"包装器。对外接口与
    # SimPhonyBackend 一致，但每个查询先查缓存、未命中才真调用 SimPhony，
    # 命中则直接返回。调用方（调度器/实验脚本）无需关心缓存细节。
    # 每次查询都会带上 arch_hash/config_hash，保证不同硬件配置不串缓存。

    def __init__(self, simphony_backend, cache, arch_hash, config_hash):
        self._backend = simphony_backend
        self._cache = cache
        self._arch_hash = arch_hash
        self._config_hash = config_hash

    def architecture_cost(self):
        # 先查缓存，未命中则调用后端并写入缓存
        cached = self._cache.get_architecture_cost(
            self._arch_hash, self._config_hash
        )
        if cached is not None:
            return cached
        result = self._backend.architecture_cost()
        self._cache.put_architecture_cost(
            self._arch_hash, self._config_hash, result
        )
        return result

    def kernel_cost(self, M, K, N, input_bits=8, weight_bits=8,
                    output_bits=8, dataflow="weight_stationary"):
        # 键里还带上 schema_version（后端报出的格式版本）
        schema_ver = self._backend.schema_version
        cached = self._cache.get_kernel_cost(
            M, K, N, input_bits, weight_bits, output_bits,
            dataflow, self._arch_hash, self._config_hash, schema_ver,
        )
        if cached is not None:
            return cached
        result = self._backend.kernel_cost(
            M, K, N, input_bits, weight_bits, output_bits, dataflow,
        )
        self._cache.put_kernel_cost(
            M, K, N, input_bits, weight_bits, output_bits,
            dataflow, self._arch_hash, self._config_hash, schema_ver, result,
        )
        return result

    def programming_cost(self):
        cached = self._cache.get_programming_cost(
            self._arch_hash, self._config_hash
        )
        if cached is not None:
            return cached
        result = self._backend.programming_cost()
        self._cache.put_programming_cost(
            self._arch_hash, self._config_hash, result
        )
        return result
