import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
)

PLOT_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
}


@st.cache_data
def load_data():
    return pd.read_csv("trained_customer_segments.csv")


@st.cache_resource
def load_model():
    return joblib.load("customer_segmentation_model.joblib")

df = load_data()
bundle = load_model()
pipeline = bundle["pipeline"]
features = bundle["features"]
segment_map = bundle["segment_map"]

st.title("Customer Segmentation & Purchasing Behavior Dashboard")
st.caption("Interactive K-Means clustering analysis across 100,000 e-commerce customers.")

# --- Sidebar Filters ---
st.sidebar.header("Filter Customers")

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=sorted(df["SegmentName"].unique()),
    default=sorted(df["SegmentName"].unique()),
)
selected_countries = st.sidebar.multiselect(
    "Country",
    options=sorted(df["Country"].unique()),
    default=sorted(df["Country"].unique()),
)
selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique()),
)
selected_devices = st.sidebar.multiselect(
    "Device Type",
    options=sorted(df["DeviceType"].unique()),
    default=sorted(df["DeviceType"].unique()),
)

filtered_df = df[
    df["SegmentName"].isin(selected_segments)
    & df["Country"].isin(selected_countries)
    & df["Category"].isin(selected_categories)
    & df["DeviceType"].isin(selected_devices)
]

if filtered_df.empty:
    st.warning("No records match your current filter selection.")
    st.stop()

# --- Top KPI Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Customers", f"{len(filtered_df):,}")
col2.metric("Avg Order Spend", f"${filtered_df['TotalAmount'].mean():,.2f}")
col3.metric("Avg Session Duration", f"{filtered_df['SessionDuration'].mean():.1f} min")
col4.metric("Avg Effective Discount", f"{filtered_df['EffectiveDiscountRate'].mean() * 100:.1f}%")

st.divider()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(
    ["Cluster Visualizations", "Purchasing Behavior", "Predict New Customer"]
)

# === TAB 1: Cluster Visualizations ===
with tab1:
    st.subheader("Cluster Projections & Centroids")
    sample_size = st.slider("Points to display", 1000, 10000, 4000, step=1000)
    sample_df = filtered_df.sample(n=min(sample_size, len(filtered_df)), random_state=42).copy()

    # 2D PCA Projection
    X_scaled = pipeline.named_steps["scaler"].transform(sample_df[features])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    sample_df["PC1"] = coords[:, 0]
    sample_df["PC2"] = coords[:, 1]

    # Compute cluster centroids
    centroids_pca = sample_df.groupby("SegmentName", as_index=False)[["PC1", "PC2"]].mean()
    centroids_spend = filtered_df.groupby("SegmentName", as_index=False)[
        ["SessionDuration", "TotalAmount"]
    ].mean()

    c1, c2 = st.columns(2)

    with c1:
        fig_pca = px.scatter(
            sample_df,
            x="PC1",
            y="PC2",
            color="SegmentName",
            hover_data=["TotalAmount", "Quantity", "EffectiveDiscountRate", "SessionDuration"],
            title=f"2D PCA Cluster Map ({pca.explained_variance_ratio_.sum() * 100:.1f}% Variance)",
            opacity=0.6,
        )
        fig_pca.update_traces(marker=dict(size=6))
        fig_pca.add_trace(
            go.Scatter(
                x=centroids_pca["PC1"],
                y=centroids_pca["PC2"],
                mode="markers",
                name="Centroids",
                text=centroids_pca["SegmentName"],
                hovertemplate="<b>Centroid: %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                marker=dict(
                    color="black",
                    symbol="x",
                    size=14,
                    line=dict(width=2, color="white"),
                ),
            )
        )
        fig_pca.update_layout(dragmode=False)
        fig_pca.update_xaxes(fixedrange=True)
        fig_pca.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_pca, use_container_width=True, config=PLOT_CONFIG)

    with c2:
        fig_spend = px.scatter(
            sample_df,
            x="SessionDuration",
            y="TotalAmount",
            color="SegmentName",
            hover_data=["Quantity", "EffectiveDiscountRate", "Category", "Country"],
            title="Session Duration vs. Total Spend by Segment",
            labels={"SessionDuration": "Session Duration (min)", "TotalAmount": "Total Spend ($)"},
            opacity=0.6,
        )
        fig_spend.update_traces(marker=dict(size=6))
        fig_spend.add_trace(
            go.Scatter(
                x=centroids_spend["SessionDuration"],
                y=centroids_spend["TotalAmount"],
                mode="markers",
                name="Centroids",
                text=centroids_spend["SegmentName"],
                hovertemplate="<b>Centroid: %{text}</b><br>Avg Session: %{x:.1f} min<br>Avg Spend: $%{y:,.2f}<extra></extra>",
                marker=dict(
                    color="black",
                    symbol="x",
                    size=14,
                    line=dict(width=2, color="white"),
                ),
            )
        )
        fig_spend.update_layout(dragmode=False)
        fig_spend.update_xaxes(fixedrange=True)
        fig_spend.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_spend, use_container_width=True, config=PLOT_CONFIG)

    st.subheader("Segment Profile Summary")
    summary_df = (
        filtered_df.groupby("SegmentName", as_index=False)
        .agg(
            Customers=("UserID", "count"),
            AvgSpend=("TotalAmount", "mean"),
            AvgQty=("Quantity", "mean"),
            AvgEffDiscount=("EffectiveDiscountRate", "mean"),
            AvgSessionMin=("SessionDuration", "mean"),
            AvgReview=("ReviewScore", "mean"),
        )
        .round(2)
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# === TAB 2: Purchasing Behavior ===
with tab2:
    b1, b2 = st.columns(2)

    with b1:
        cat_spend = (
            filtered_df.groupby(["Category", "SegmentName"], as_index=False)["TotalAmount"]
            .mean()
            .round(2)
        )
        fig_cat = px.bar(
            cat_spend,
            x="Category",
            y="TotalAmount",
            color="SegmentName",
            barmode="group",
            title="Average Spend by Product Category & Segment",
            labels={"TotalAmount": "Avg Spend ($)"},
        )
        fig_cat.update_layout(dragmode=False)
        fig_cat.update_xaxes(fixedrange=True)
        fig_cat.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_cat, use_container_width=True, config=PLOT_CONFIG)

    with b2:
        dev_counts = (
            filtered_df.groupby(["DeviceType", "SegmentName"], as_index=False)["UserID"]
            .count()
            .rename(columns={"UserID": "Customers"})
        )
        fig_dev = px.bar(
            dev_counts,
            x="DeviceType",
            y="Customers",
            color="SegmentName",
            barmode="stack",
            title="Customer Distribution by Device & Segment",
        )
        fig_dev.update_layout(dragmode=False)
        fig_dev.update_xaxes(fixedrange=True)
        fig_dev.update_yaxes(fixedrange=True)
        st.plotly_chart(fig_dev, use_container_width=True, config=PLOT_CONFIG)

    # Monthly Trend
    monthly = (
        filtered_df.groupby("PurchaseMonth", as_index=False)["TotalAmount"]
        .sum()
        .sort_values("PurchaseMonth")
    )
    fig_month = px.line(
        monthly,
        x="PurchaseMonth",
        y="TotalAmount",
        markers=True,
        title="Total Revenue Trend by Purchase Month",
        labels={"PurchaseMonth": "Month", "TotalAmount": "Total Revenue ($)"},
    )
    fig_month.update_layout(dragmode=False)
    fig_month.update_xaxes(fixedrange=True)
    fig_month.update_yaxes(fixedrange=True)
    st.plotly_chart(fig_month, use_container_width=True, config=PLOT_CONFIG)

# === TAB 3: Predict New Customer ===
with tab3:
    st.subheader("Classify a New Customer into a Segment")

    p1, p2 = st.columns(2)
    with p1:
        in_spend = st.number_input("Total Order Spend ($)", min_value=10.0, max_value=5000.0, value=1200.0, step=50.0)
        in_qty = st.slider("Quantity Purchased", min_value=1, max_value=4, value=2)
    with p2:
        in_disc = st.slider("Effective Discount Rate (0.00 - 0.50)", min_value=0.0, max_value=0.50, value=0.10, step=0.01)
        in_sess = st.slider("Session Duration (Minutes)", min_value=5.0, max_value=120.0, value=60.0, step=1.0)

    if st.button("Predict Segment", type="primary"):
        new_row = pd.DataFrame(
            [
                {
                    "TotalAmount": in_spend,
                    "Quantity": in_qty,
                    "EffectiveDiscountRate": in_disc,
                    "SessionDuration": in_sess,
                }
            ]
        )
        cluster_id = int(pipeline.predict(new_row[features])[0])
        predicted_segment = segment_map[cluster_id]

        st.success(f"**Predicted Segment:** {predicted_segment} (Cluster {cluster_id})")

        recommendations = {
            "High-Value Bulk Buyers": "Offer VIP loyalty perks, early product access, and free express shipping.",
            "Discount Hunters": "Target with flash sales, bundle coupons, and seasonal promo codes.",
            "Engaged Browsers": "Trigger cart-abandonment reminders or personalized product recommendations.",
            "Quick Casual Shoppers": "Streamline one-click checkout and cross-sell low-friction add-ons.",
        }
        st.info(f"**Suggested Action:** {recommendations.get(predicted_segment, '')}")