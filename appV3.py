import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1. 網頁初始化配置
st.set_page_config(page_title="F121 製程最佳化控制系統", layout="wide")
st.title("🏭 F121 天然氣最低消耗與製程控制最佳化系統")

# 2. 檔案上傳元件
uploaded_file = st.file_uploader("請上傳您的 F121 歷史數據 Excel 檔 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 讀取 Excel 檔案，跳過第二行 Tag 行
        df = pd.read_excel(uploaded_file, skiprows=[1])
        df.columns = df.columns.str.strip()
        
        feature_uncontrollable = ['DT operation', 'C141 operation']
        feature_controllable = ['F121 CLO circulation flow', 'F121outlet temperature', 'F121 Oxygen content %']
        all_features = feature_uncontrollable + feature_controllable
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
            
            # 訓練模型（使用快取避免重複計算）
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
            
            # 3. 側邊欄：不可控變數輸入 (調整時，下方結果會跟著動)
            st.sidebar.header("📋 當前不可控製程條件")
            input_dt = st.sidebar.slider("DT operation 稼動率", bounds_config['dt_min'], bounds_config['dt_max'], (bounds_config['dt_min'] + bounds_config['dt_max'])/2)
            input_c141 = st.sidebar.slider("C141 operation 稼動率", bounds_config['c141_min'], bounds_config['c141_max'], (bounds_config['c141_min'] + bounds_config['c141_max'])/2)

            # 4. 主畫面：設定可控參數操作限制
            st.header("⚙️ 設定可控參數的操作安全限制範圍 (Safety Bounds)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### CLO Circulation Flow")
                clo_min = st.number_input("安全下限", value=bounds_config['clo_min'], key="clo_min")
                clo_max = st.number_input("安全上限", value=bounds_config['clo_max'], key="clo_max")
            with col2:
                st.markdown("### Outlet Temperature")
                temp_min = st.number_input("安全下限 (°C)", value=bounds_config['out_temp_min'], key="t_min")
                temp_max = st.number_input("安全上限 (°C)", value=bounds_config['out_temp_max'], key="t_max")
            with col3:
                st.markdown("### Oxygen Content %")
                ox_min = st.number_input("安全下限 (%)", value=bounds_config['ox_min'], key="ox_min")
                ox_max = st.number_input("安全上限 (%)", value=bounds_config['ox_max'], key="ox_max")

            # 5. 最佳化核心尋優計算 (改用網格生成法，避開隨機森林梯度為 0 的問題)
            if st.button("🚀 開始計算最低天然氣消耗控制策略", type="primary"):
                with st.spinner("正在全局搜索最省能的操作組合..."):
                    # 在使用者設定的範圍內，各自切出 25 個均勻點 (共 25^3 = 15,625 種組合，矩陣平行運算只要不到 0.5 秒)
                    grid_clo = np.linspace(clo_min, clo_max, 25)
                    grid_temp = np.linspace(temp_min, temp_max, 25)
                    grid_ox = np.linspace(ox_min, ox_max, 25)
                    
                    # 建立網格矩陣
                    c_mesh, t_mesh, o_mesh = np.meshgrid(grid_clo, grid_temp, grid_ox)
                    
                    # 展平成預報矩陣形式
                    flat_clo = c_mesh.ravel()
                    flat_temp = t_mesh.ravel()
                    flat_ox = o_mesh.ravel()
                    
                    # 填充目前固定的不可控變數
                    flat_dt = np.full_like(flat_clo, input_dt)
                    flat_c141 = np.full_like(flat_clo, input_c141)
                    
                    # 組合出所有測試特徵集
                    test_features = np.column_stack([flat_dt, flat_c141, flat_clo, flat_temp, flat_ox])
                    
                    # 批量預測所有組合的 NG 消耗量
                    pred_ng_all = model_ng.predict(test_features)
                    
                    # 找出 NG 消耗量最小的那一組索引
                    best_idx = np.argmin(pred_ng_all)
                    
                    # 提取最佳控制參數與對應的最低 NG 消耗
                    opt_clo = flat_clo[best_idx]
                    opt_temp = flat_temp[best_idx]
                    opt_ox = flat_ox[best_idx]
                    min_ng_consumption = pred_ng_all[best_idx]
                    
                    # 使用這一組最佳參數來預測 C122 的底部溫度
                    best_feature_row = np.array([[input_dt, input_c141, opt_clo, opt_temp, opt_ox]])
                    predicted_c122_temp = model_temp.predict(best_feature_row)[0]
                    
                    # 6. 結果呈現
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
                    
    except Exception as e:
        st.error(f"❌ 讀取 Excel 檔案或計算時發生錯誤: {e}")
else:
    st.info("💡 請在上方直接上傳您的原始 F121 數據 Excel (.xlsx) 檔案以啟動系統。")
