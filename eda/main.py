import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eda.agents.overview import analyze_overview
from eda.agents.quality import analyze_quality
from eda.agents.profiling import analyze_profiling
from eda.agents.visuals import generate_visuals
from eda.agents.features import analyze_features
from eda.agents.ml_readiness import analyze_ml_readiness
from eda.agents.insights import generate_draft_analysis
from eda.agents.auditor import audit_draft_analysis
from eda.agents.report import build_final_reports

def main():
    print("=" * 60)
    print("🚀 INICIANDO PIPELINE AGÉNTICO AUTÓNOMO DE EDA")
    print("=" * 60)
    
    # Paths
    trans_path = "data/transactions.csv"
    stores_path = "data/stores.csv"
    cal_path = "data/calendar.csv"
    
    stats_json_path = "outputs/stats_raw.json"
    draft_md_path = "outputs/draft_analysis.md"
    audit_log_path = "outputs/audit_log.md"
    figures_dir = "figures"
    report_md_path = "reports/EDA_Report.md"
    report_html_path = "reports/EDA_Report.html"
    
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # -------------------------------------------------------------
    # FASE 1: Extracción, Profiling Técnico y Visuales
    # -------------------------------------------------------------
    print("\n📦 [FASE 1] Cargando datasets...")
    df_trans = pd.read_csv(trans_path)
    df_stores = pd.read_csv(stores_path)
    df_cal = pd.read_csv(cal_path)
    
    print("🔍 [FASE 1.1] Ejecutando Overview & Integrity Agent...")
    overview_data = analyze_overview(df_trans, df_stores, df_cal)
    
    print("🧹 [FASE 1.2] Ejecutando Data Quality & Outliers Agent...")
    quality_data = analyze_quality(df_trans, df_stores, df_cal)
    
    print("📊 [FASE 1.3] Ejecutando Profiling Estadístico y Métricas de Negocio...")
    profiling_data = analyze_profiling(df_trans, df_stores, df_cal)
    
    print("🎨 [FASE 1.4] Ejecutando Visual Agent (Generando gráficos 300 DPI)...")
    visual_files = generate_visuals(df_trans, df_stores, df_cal, output_dir=figures_dir)
    print(f"   ✓ Gráficos generados: {len(visual_files)} en '{figures_dir}/'")
    
    # FASE 2 sub-agentes de síntesis
    print("💡 [FASE 2.1] Ejecutando Feature Engineering Advisor...")
    features_data = analyze_features(df_trans, df_stores, df_cal)
    
    print("🤖 [FASE 2.2] Ejecutando ML Readiness Agent...")
    ml_data = analyze_ml_readiness(df_trans, df_stores, df_cal)
    
    # Guardar stats_raw.json (Única Fuente de la Verdad)
    stats_raw = {
        "overview": overview_data,
        "quality": quality_data,
        "profiling": profiling_data,
        "features": features_data,
        "ml_readiness": ml_data
    }
    
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_raw, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Guardado Fuente Única de la Verdad en: '{stats_json_path}'")
    
    # -------------------------------------------------------------
    # FASE 2: Síntesis de Borrador
    # -------------------------------------------------------------
    print("\n📝 [FASE 2.3] Generando borrador de análisis ('outputs/draft_analysis.md')...")
    generate_draft_analysis(stats_json_path, output_path=draft_md_path)
    print(f"   ✓ Borrador generado en: '{draft_md_path}'")
    
    # -------------------------------------------------------------
    # FASE 3: Auditoría Fáctica (Fact-Checker)
    # -------------------------------------------------------------
    print("\n🔎 [FASE 3] Ejecutando Auditor Fáctico (Fact-Checker Agent)...")
    audit_res = audit_draft_analysis(stats_json_path, draft_md_path, audit_log_path=audit_log_path)
    print(f"   ✓ Auditoría completada: {audit_res['passed_count']} pasadas, {audit_res['failed_count']} discrepancias.")
    print(f"   ✓ Registro de auditoría guardado en: '{audit_log_path}'")
    
    # -------------------------------------------------------------
    # FASE 4: Report Builder
    # -------------------------------------------------------------
    print("\n📄 [FASE 4] Ejecutando Report Builder Agent...")
    report_res = build_final_reports(stats_json_path, figures_dir=figures_dir, output_md=report_md_path, output_html=report_html_path)
    print(f"   ✓ Reporte Markdown generado en: '{report_res['output_md']}'")
    print(f"   ✓ Reporte HTML responsivo autocontenido generado en: '{report_res['output_html']}'")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE CON CERO INTERACCIÓN Y 100% REPRODUCIBLE")
    print("=" * 60)

if __name__ == "__main__":
    main()
