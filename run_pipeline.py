#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline.py — CulturLand-Check 主流水线
=============================================
一键运行全部或指定阶段:
  Stage 1: 数据质量检测与修复        (7_quality_check.py)
  Stage 2: Schema约束实体关系抽取     (8_csge_extraction.py)
  Stage 3: 知识图谱构建               (9_build_kg.py)
  Stage 4: 三维指标计算               (10_indicator_engine.py)
  Stage 5: TD-GRAG 诊断              (11_td_grag_diagnosis.py)
  Stage 6: Baseline 方法运行          (13_baselines.py)
  Stage 7: 评估汇总                   (12_evaluation.py)
  Stage 8: 论文图表生成               (14_visualize.py)

用法:
  python run_pipeline.py                     # 运行全部
  python run_pipeline.py --stage 1 3 5       # 仅运行 Stage 1,3,5
  python run_pipeline.py --from-stage 4      # 从 Stage 4 开始
  python run_pipeline.py --no-llm            # 跳过需要 LLM 的步骤
  python run_pipeline.py --dry-run           # 仅打印执行计划
"""

import sys, os, time, argparse, logging, json, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("pipeline")


# ================================================================
# 流水线阶段定义
# ================================================================

STAGES = [
    {
        "id": 1,
        "name": "数据质量检测与修复",
        "script": "src/7_quality_check.py",
        "args": [],
        "needs_llm": True,
        "no_llm_args": ["--no-llm"],
        "description": "Algorithm 3: DQ-1~DQ-6 数据修复策略",
    },
    {
        "id": 2,
        "name": "Schema约束实体关系抽取 (CSGE)",
        "script": "src/8_csge_extraction.py",
        "args": [],
        "needs_llm": True,
        "no_llm_args": ["--no-llm"],
        "description": "Algorithm 1: 二阶段抽取 + 空间语义校验",
    },
    {
        "id": 3,
        "name": "知识图谱构建",
        "script": "src/9_build_kg.py",
        "args": [],
        "needs_llm": False,
        "description": "NetworkX MultiDiGraph + Neo4j CSV 导出",
    },
    {
        "id": 4,
        "name": "三维指标计算",
        "script": "src/10_indicator_engine.py",
        "args": [],
        "needs_llm": False,
        "description": "供给侧/需求侧/质量侧指标 + 综合健康指数",
    },
    {
        "id": 5,
        "name": "TD-GRAG 诊断",
        "script": "src/11_td_grag_diagnosis.py",
        "args": [],
        "needs_llm": True,
        "description": "Algorithm 2: 三维子图检索 + 证据链 + Chain-of-Rationale",
    },
    {
        "id": 6,
        "name": "Baseline 方法运行",
        "script": "src/13_baselines.py",
        "args": ["--baseline", "all", "--ablation", "all"],
        "needs_llm": True,
        "no_llm_args": ["--baseline", "rule_only", "--no-llm"],
        "description": "B1~B4 对照方法 + 消融实验",
    },
    {
        "id": 7,
        "name": "评估汇总",
        "script": "src/12_evaluation.py",
        "args": ["--mode", "full"],
        "needs_llm": False,
        "description": "RQ1~RQ3 指标计算 + 统计检验",
    },
    {
        "id": 8,
        "name": "论文图表生成",
        "script": "src/14_visualize.py",
        "args": [],
        "needs_llm": False,
        "description": "Fig.4~6 + Tab.1~3 + LaTeX",
    },
]


def run_stage(stage: dict, no_llm: bool = False, dry_run: bool = False) -> bool:
    """运行单个阶段"""
    stage_id = stage["id"]
    script_path = ROOT / stage["script"]

    if not script_path.exists():
        log.error(f"  脚本不存在: {script_path}")
        return False

    # 构造命令
    cmd = [sys.executable, str(script_path)]
    if no_llm and stage.get("needs_llm"):
        cmd.extend(stage.get("no_llm_args", stage["args"]))
    else:
        cmd.extend(stage["args"])

    log.info(f"\n{'='*60}")
    log.info(f"Stage {stage_id}: {stage['name']}")
    log.info(f"  脚本: {stage['script']}")
    log.info(f"  命令: {' '.join(cmd)}")
    log.info(f"  说明: {stage['description']}")
    log.info(f"{'='*60}")

    if dry_run:
        log.info("  [DRY RUN] 跳过实际执行")
        return True

    start_time = time.time()
    # 注入 PYTHONIOENCODING=utf-8，避免子进程在 Windows GBK 终端下 emoji 编码错误
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=False,
            text=True,
            timeout=3600,  # 1小时超时
            env=sub_env,
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            log.info(f"  [OK] Stage {stage_id} 完成 (耗时 {elapsed:.1f}s)")
            return True
        else:
            log.error(f"  [FAIL] Stage {stage_id} 失败 (返回码 {result.returncode}, 耗时 {elapsed:.1f}s)")
            return False

    except subprocess.TimeoutExpired:
        log.error(f"  [TIMEOUT] Stage {stage_id} 超时 (>3600s)")
        return False
    except Exception as e:
        log.error(f"  [ERROR] Stage {stage_id} 异常: {e}")
        return False


def run_pipeline(stages_to_run: list, no_llm: bool, dry_run: bool):
    """运行流水线"""
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         CulturLand-Check Pipeline Runner                ║")
    log.info("║   融合KG+RAG+LLM的城市文化用地体检方法                 ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  工作目录: {ROOT}")
    log.info(f"  待运行阶段: {[s['id'] for s in stages_to_run]}")
    log.info(f"  No-LLM 模式: {no_llm}")
    log.info(f"  Dry-Run: {dry_run}")

    results = {}
    total_start = time.time()

    for stage in stages_to_run:
        if no_llm and stage.get("needs_llm") and "no_llm_args" not in stage:
            log.info(f"\n  ⏭️ 跳过 Stage {stage['id']} ({stage['name']}) — 需要 LLM")
            results[stage["id"]] = "skipped"
            continue

        success = run_stage(stage, no_llm=no_llm, dry_run=dry_run)
        results[stage["id"]] = "success" if success else "failed"

        if not success and not dry_run:
            log.warning(f"  Stage {stage['id']} 失败, 继续执行后续阶段...")

    total_elapsed = time.time() - total_start

    # 打印总结
    log.info("\n" + "="*60)
    log.info("📋 流水线执行总结")
    log.info("="*60)
    for stage in stages_to_run:
        status = results.get(stage["id"], "unknown")
        icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(status, "❓")
        log.info(f"  {icon} Stage {stage['id']}: {stage['name']} — {status}")
    log.info(f"\n  总耗时: {total_elapsed:.1f}s")

    # 保存执行记录
    if not dry_run:
        record = {
            "timestamp": datetime.now().isoformat(),
            "total_seconds": round(total_elapsed, 1),
            "no_llm": no_llm,
            "results": {str(k): v for k, v in results.items()},
        }
        record_dir = ROOT / "artifacts"
        record_dir.mkdir(exist_ok=True)
        record_path = record_dir / "pipeline_run_log.json"
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        log.info(f"  执行记录: {record_path}")

    failed = [sid for sid, status in results.items() if status == "failed"]
    if failed:
        log.warning(f"\n  ⚠️ 失败阶段: {failed}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CulturLand-Check 主流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py                     # 运行全部
  python run_pipeline.py --stage 1 3 5       # 仅运行 Stage 1,3,5
  python run_pipeline.py --from-stage 4      # 从 Stage 4 开始
  python run_pipeline.py --no-llm            # 仅运行无需 LLM 的步骤
  python run_pipeline.py --dry-run           # 仅打印执行计划
        """
    )
    parser.add_argument("--stage", nargs="+", type=int, default=None,
                        help="指定运行的阶段编号 (1-8)")
    parser.add_argument("--from-stage", type=int, default=None,
                        help="从指定阶段开始运行")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过或降级需要 LLM 的步骤")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印执行计划, 不实际运行")
    args = parser.parse_args()

    # 确定要运行的阶段
    if args.stage:
        stage_ids = set(args.stage)
        stages_to_run = [s for s in STAGES if s["id"] in stage_ids]
    elif args.from_stage:
        stages_to_run = [s for s in STAGES if s["id"] >= args.from_stage]
    else:
        stages_to_run = STAGES

    if not stages_to_run:
        log.error("未找到匹配的阶段")
        sys.exit(1)

    exit_code = run_pipeline(stages_to_run, no_llm=args.no_llm, dry_run=args.dry_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
