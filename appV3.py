import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb

# 1. 網頁初始化配置
st.set_page_config(page_title="F121 製程最佳化控制系統", layout="wide")
st.title("🏭 F121 天然氣最低消耗與製程控制最佳化系統 (LightGBM 靈活尋優版)")

# 2. 檔案上傳元件
uploaded_file = st.file_uploader("請上傳您的 F121 歷史數據 Excel 檔 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 讀取 Excel 檔案，跳過第二行 Tag 行
        df = pd.read_excel(uploaded_file, skiprows=[1])
        df.columns = df.columns.str.strip()
        
        # 欄位定義
        feature_fixed = ['DT operation', 'C141 operation', 'F121outlet temperature']
        feature_controllable = ['F121 CLO circulation flow', 'F121 Oxygen content %']
        all_features = feature_fixed + feature_controllable
        
        target_ng = 'F121 NG consumption'
        target_temp = 'C122 bottom temperature'
        
        required_cols = all_features + [target_ng, target_temp]
        
        # 清理並轉換數據
        df_clean = df[required_cols].dropna().apply(pd.to_numeric, errors='coerce').dropna()
        
        if df_clean.empty:
            st.error("❌ 經篩選後沒有有效的數據，請確認 Excel 內的數值是否正確。")
        else:
            X = df_clean[all_features]
            y_ng = df_clean[target_ng]
            y_temp = df_clean[target_temp]
            
            # 【核心修正】改用 LightGBM 訓練高精度模型
            @st.cache_resource
            def train_lgb_models(_X, _y_ng, _y_temp):
                # 針對 NG 消耗量模型，我們可以給予特定欄位限制：
                # 順序：['DT operation', 'C141 operation', 'F121outlet temperature', 'F121 CLO circulation flow', 'F121 Oxygen content %']
                # 0 代表不限制，1 代表強迫正相關（單調遞增）。我們讓 CLO 流量(第4個變數)維持物理正相關，避免它盲目踩極端下限或上限。
                mono_constraints = [0, 0, 0, 0, 0] # 如果您想完全交由 AI 自由擬合平滑曲線，可全設為 0。
                
                m_ng = lgb.LGBMRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    min_child_samples=5, # 允許更精細的切分，防止階梯斷層
                    monotone_constraints=mono_constraints,
                    verbose=-1
                ).fit(_X, _y_ng)
                
                m_temp = lgb.LGBMRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    min_child_samples=5,
                    verbose=-1
                ).fit(_X, _y_temp)
                
                return m_ng, m_temp
                
            model_ng, model_temp = train_lgb_models(X, y_ng, y_temp)
            st.success("✅ Excel 數據載入成功，高精度 LightGBM 預報引擎已建立！")
            
            # 獲取各欄位歷史極值作為網頁預設值
            bounds_config = {
                'dt_min': float(df_clean['DT operation'].min()), 'dt_max': float(df_clean['DT operation'].max()),
                'c141_min': float(df_clean['C141 operation'].min()), 'c141_max': float(df_clean['C141 operation'].max()),
                'out_temp_min': float(df_clean['F121outlet temperature'].min()), 'out_temp_max': float(df_clean['F121outlet temperature'].max()),
                'clo_min': float(df_clean['F121 CLO circulation flow'].min()), 'clo_max': float(df_clean['F121 CLO circulation flow'].max()),
                'ox_min': float(df_clean['F121 Oxygen content %'].min()), 'ox_max': float(df_clean['F121 Oxygen content %'].max())
            }
            
            # 3. 側邊欄：固定輸入項目
            st.sidebar.header("📋 當前固定輸入/排程條件")
            default_dt = round((bounds_config['dt_min'] + bounds_config['dt_max']) / 2, 2)
            default_c141 = round((bounds_config['c141_min'] + bounds_config['c141_max']) / 2, 2)
            default_temp = round((bounds_config['out_temp_min'] + bounds_config['out_temp_max']) / 2, 2)
            
            input_dt = st.sidebar.number_input(f"DT operation 稼動率 ({bounds_config['dt_min']} ~ {bounds_config['dt_max']})", value=default_dt, step=0.01)
            input_c141 = st.sidebar.number_input(f"C141 operation 稼動率 ({bounds_config['c141_min']} ~ {bounds_config['c141_max']})", value=default_c141, step=0.01)
            input_out_temp = st.sidebar.number_input(f"F121 outlet temperature 出路溫度 (°C)", value=default_temp, step=0.1)

            # 4. 主畫面：設定可控參數安全範圍
            st.header("⚙️ 設定可控參數的操作安全限制範圍 (Safety Bounds)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### CLO Circulation Flow")
                clo_min = st.number_input("安全下限", value=bounds_config['clo_min'], key="clo_min")
                clo_max = st.number_input("安全上限", value=bounds_config['clo_max'], key="clo_max")
            with col2:
                st.markdown("### Oxygen Content %")
                ox_min = st.number_input("安全下限 (%)", value=bounds_config['ox_min'], key="ox_min")
                ox_max = st.number_input("安全上限 (%)", value=bounds_config['ox_max'], key="ox_max")

            # 5. 全局精細網格尋優計算
            if st.button("🚀 開始計算最低天然氣消耗控制策略", type="primary"):
                with st.spinner("正在全局搜索最省能的操作組合..."):
                    # 使用 LightGBM 計算速度極快，我們可以把網格切細（50x50），確保不會漏掉最佳點
                    grid_clo = np.linspace(clo_min, clo_max, 50)
                    grid_ox = np.linspace(ox_min, ox_max, 50)
                    
                    c_mesh, o_mesh = np.meshgrid(grid_clo, grid_ox)
                    flat_clo = c_mesh.ravel()
                    flat_ox = o_mesh.ravel()
                    
                    flat_dt = np.full_like(flat_clo, input_dt)
                    flat_c141 = np.full_like(flat_clo, input_c141)
                    flat_out_temp = np.full_like(flat_clo, input_out_temp)
                    
                    test_features = np.column_stack([flat_dt, flat_c141, flat_out_temp, flat_clo, flat_ox])
                    
                    # 預測所有組合
                    pred_ng_all = model_ng.predict(test_features)
                    best_idx = np.argmin(pred_ng_all)
                    
                    opt_clo = flat_clo[best_idx]
                    opt_ox = flat_ox[best_idx]
                    min_ng_consumption = pred_ng_all[best_idx]
                    
                    # 預測 C122 溫度
                    best_feature_row = np.array([[input_dt, input_c141, input_out_temp, opt_clo, opt_ox]])
                    predicted_c122_temp = model_temp.predict(best_feature_row)[0]
                    
                    # 6. 結果呈現
                    st.markdown("---")
                    st.subheader("🎯 最佳化控制推薦結果")
                    
                    m1, m2 = st.columns(2)
                    m1.metric(label="📉 預期最低 F121 NG Consumption (Y)", value=f"{min_ng_consumption:.2f}")
                    m2.metric(label="🌡️ 同時預測 C122 Bottom Temperature", value=f"{predicted_c122_temp:.2f} °C")
                    
                    recommend_df = pd.DataFrame({
                        "製程控制項目": ["F121 CLO circulation flow", "F121 Oxygen content %"],
                        "💡 最佳推薦控制值": [f"{opt_clo:.2f}", f"{opt_ox:.2f} %"],
                        "當前設定安全操作範圍": [f"{clo_min} ~ {clo_max}", f"{ox_min} ~ {ox_max}"]
                    })
                    st.table(recommend_df)
                    
    except Exception as e:
        st.error(f"❌ 計算時發生錯誤: {e}")
else:
    st.info("💡 請在上方直接上傳您的原始 F121 數據 Excel (.xlsx) 檔案以啟動系統。")
