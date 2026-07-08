import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

# 1. 網頁初始化配置
st.set_page_config(page_title="F121 製程最佳化控制系統", layout="wide")
st.title("🏭 F121 天然氣最低消耗與製程控制最佳化系統")

# 2. 檔案上傳元件：這次真正支援您電腦裡的原始 .xlsx 檔案
uploaded_file = st.file_uploader("請上傳您的 F121 歷史數據 Excel 檔 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 【關鍵修正】使用 pd.read_excel 讀取 Excel 檔案
        # skiprows=[1] 用來跳過第二行 (也就是 TR122-11, TC121-72 那行 Tag 代碼)，保留第一行作為欄位名稱
        df = pd.read_excel(uploaded_file, skiprows=[1])
        
        # 清理欄位名稱（去除 Excel 可能產生的前後空格或隱藏字元）
        df.columns = df.columns.str.strip()
        
        # 定義變數欄位 (必須與 Excel 第一行的文字完全一模一樣)
        feature_uncontrollable = ['DT operation', 'C141 operation']
        feature_controllable = ['F121 CLO circulation flow', 'F121outlet temperature', 'F121 Oxygen content %']
        all_features = feature_uncontrollable + feature_controllable
        
        target_ng = 'F121 NG consumption'
        target_temp = 'C122 bottom temperature'
        
        # 檢查 Excel 欄位是否正確讀取
        required_cols = all_features + [target_ng, target_temp]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ 讀取成功，但 Excel 內缺少以下必要欄位，請檢查拼字是否一致：{missing_cols}")
            st.info(f"目前偵測到的 Excel 欄位有：{list(df.columns)}")
        else:
            # 移除非數值與缺失值（處理 Excel 內可能包含的空白格或文字）
            df_clean = df[required_cols].dropna().apply(pd.to_numeric, errors='coerce').dropna()
            
            if df_clean.empty:
                st.error("❌ 經篩選後沒有有效的數據，請確認 Excel 內的數值是否正確（不可包含文字文字）。")
            else:
                X = df_clean[all_features]
                y_ng = df_clean[target_ng]
                y_temp = df_clean[target_temp]
                
                # 訓練模型（使用快取避免網頁重新整理時重複計算）
                @st.cache_resource
                def train_models(_X, _y_ng, _y_temp):
                    m_ng = RandomForestRegressor(n_estimators=100, random_state=42).fit(_X, _y_ng)
                    m_temp = RandomForestRegressor(n_estimators=100, random_state=42).fit(_X, _y_temp)
                    return m_ng, m_temp
                    
                model_ng, model_temp = train_models(X, y_ng, y_temp)
                st.success("✅ Excel 數據載入成功，AI 模型已訓練完成！")
                
                # 獲取各欄位歷史極值作為網頁範圍參考
                bounds_config = {
                    'dt_min': float(df_clean['DT operation'].min()), 'dt_max': float(df_clean['DT operation'].max()),
                    'c141_min': float(df_clean['C141 operation'].min()), 'c141_max': float(df_clean['C141 operation'].max()),
                    'clo_min': float(df_clean['F121 CLO circulation flow'].min()), 'clo_max': float(df_clean['F121 CLO circulation flow'].max()),
                    'out_temp_min': float(df_clean['F121outlet temperature'].min()), 'out_temp_max': float(df_clean['F121outlet temperature'].max()),
                    'ox_min': float(df_clean['F121 Oxygen content %'].min()), 'ox_max': float(df_clean['F121 Oxygen content %'].max())
                }
                
                # 3. 側邊欄：不可控變數輸入
                st.sidebar.header("📋 當前不可控製程條件")
                input_dt = st.sidebar.slider("DT operation 稼動率", bounds_config['dt_min'], bounds_config['dt_max'], (bounds_config['dt_min'] + bounds_config['dt_max'])/2)
                input_c141 = st.sidebar.slider("C141 operation 稼動率", bounds_config['c141_min'], bounds_config['c141_max'], (bounds_config['c141_min'] + bounds_config['c141_max'])/2)

                # 4. 主畫面：設定可控參數操作限制
                st.header("⚙️ 設定可控參數的操作安全限制範圍 (Safety Bounds)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### CLO Circulation Flow")
                    clo_min = st.number_input("安全下限", value=bounds_config['clo_min'])
                    clo_max = st.number_input("安全上限", value=bounds_config['clo_max'])
                with col2:
                    st.markdown("### Outlet Temperature")
                    temp_min = st.number_input("安全下限 (°C)", value=bounds_config['out_temp_min'])
                    temp_max = st.number_input("安全上限 (°C)", value=bounds_config['out_temp_max'])
                with col3:
                    st.markdown("### Oxygen Content %")
                    ox_min = st.number_input("安全下限 (%)", value=bounds_config['ox_min'])
                    ox_max = st.number_input("安全上限 (%)", value=bounds_config['ox_max'])

                # 5. 最佳化核心尋優計算
                if st.button("🚀 開始計算最低天然氣消耗控制策略", type="primary"):
                    def objective_func(controllable_vars):
                        features = np.array([[input_dt, input_c141, controllable_vars[0], controllable_vars[1], controllable_vars[2]]])
                        return model_ng.predict(features)[0]

                    opt_bounds = [(clo_min, clo_max), (temp_min, temp_max), (ox_min, ox_max)]
                    initial_guess = [(clo_min + clo_max) / 2, (temp_min + temp_max) / 2, (ox_min + ox_max) / 2]
                    
                    res = minimize(objective_func, initial_guess, method='SLSQP', bounds=opt_bounds)
                    
                    if res.success:
                        opt_clo, opt_temp, opt_ox = res.x
                        min_ng_consumption = res.fun
                        
                        final_features = np.array([[input_dt, input_c141, opt_clo, opt_temp, opt_ox]])
                        predicted_c122_temp = model_temp.predict(final_features)[0]
                        
                        st.markdown("---")
                        st.subheader("🎯 最佳化控制推薦結果")
                        
                        m1, m2 = st.columns(2)
                        m1.metric(label="📉 預期最低 F121 NG Consumption (Y)", value=f"{min_ng_consumption:.2f}")
                        m2.metric(label="🌡️ 同時預測 C122 Bottom Temperature", value=f"{predicted_c122_temp:.2f} °C")
                        
                        recommend_df = pd.DataFrame({
                            "製程控制項目": ["F121 CLO circulation flow", "F121 outlet temperature", "F121 Oxygen content %"],
                            "💡 最佳推薦控制值": [f"{opt_clo:.2f}", f"{opt_temp:.2f}", f"{opt_ox:.2f} %"],
                            "當前設定安全操作範圍": [f"{clo_min} ~ {clo_max}", f"{temp_min} ~ {temp_max}", f"{ox_min} ~ {ox_max}"]
                        })
                        st.table(recommend_df)
                    else:
                        st.error("❌ 優化演算法未收斂，請適度放寬安全控制範圍再試一次。")
    except Exception as e:
        st.error(f"❌ 讀取 Excel 檔案時發生未知錯誤: {e}")
else:
    st.info("💡 請在上方直接上傳您的原始 F121 數據 Excel (.xlsx) 檔案以啟動系統。")