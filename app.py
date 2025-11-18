# app.py

import streamlit as st
import numpy as np
import pickle
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(
    page_title="🎨 Art Style Recommender",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
    <style>
    .title { text-align: center; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; font-size: 1.2rem; color: #888; margin-bottom: 2rem; }
    .card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .rec-card { border: 2px solid #ddd; padding: 1rem; border-radius: 0.5rem; text-align: center; }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODELS (Cached for performance)
# ============================================================================

@st.cache_resource
def load_models():
    """Load all pre-trained models and data"""

    model_dir = "models"

    try:
        # Load PCA and K-Means
        pca = pickle.load(open(f'{model_dir}/pca_2d.pkl', 'rb'))
        kmeans = pickle.load(open(f'{model_dir}/kmeans_model.pkl', 'rb'))

        # Load data arrays
        cluster_labels = np.load(f'{model_dir}/cluster_labels.npy')
        features_2d = np.load(f'{model_dir}/features_2d.npy')
        image_paths = pickle.load(open(f'{model_dir}/image_paths.pkl', 'rb'))
        features = np.load(f'{model_dir}/features.npy')

        # Load ResNet50 feature extractor
        resnet = models.resnet50(pretrained=True)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        resnet.eval()

        return {
            'pca': pca,
            'kmeans': kmeans,
            'cluster_labels': cluster_labels,
            'features_2d': features_2d,
            'image_paths': image_paths,
            'features': features,
            'resnet': resnet
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


models_data = load_models()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_features(image):
    """Extract ResNet features from image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = models_data['resnet'](img_tensor).squeeze().numpy()

    return features


def get_recommendations(features, cluster, top_n=5):
    """Get top N similar artworks from same cluster"""

    cluster_mask = models_data['cluster_labels'] == cluster
    cluster_indices = np.where(cluster_mask)[0]

    # Calculate cosine similarity
    similarities = cosine_similarity(
        [features],
        models_data['features'][cluster_indices]
    )[0]

    # Get top N
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        actual_idx = cluster_indices[idx]
        recommendations.append({
            'rank': rank,
            'image_path': models_data['image_paths'][actual_idx],
            'similarity': similarities[idx]
        })

    return recommendations


def get_cluster_name(cluster_id):
    """Map cluster ID to art style name"""
    cluster_names = {
        0: "🖼️ Classical & Portrait",
        1: "🌈 Modern & Abstract",
        2: "🌳 Landscape & Nature"
    }
    return cluster_names.get(cluster_id, f"Cluster {cluster_id}")


# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("""
    <h1 class='title'>🎨 AI Art Style Recommender</h1>
    <p class='subtitle'>Upload an artwork and discover similar paintings</p>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio(
    "📍 Navigation",
    ["🚀 Upload & Predict", "📊 Explore Clusters", "🔍 Visualization", "ℹ️ About"]
)

# ============================================================================
# PAGE 1: UPLOAD & PREDICT
# ============================================================================

if page == "🚀 Upload & Predict":
    st.subheader("Upload Your Artwork")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📤 Your Upload")
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            key='uploader'
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Your Upload", use_column_width=True)

            # Extract features
            with st.spinner("🔄 Analyzing artwork..."):
                features = extract_features(image)
                cluster = int(models_data['kmeans'].predict([features])[0])

            # Display results
            st.markdown("---")
            st.success(f"✅ Art Style Detected: **{get_cluster_name(cluster)}**")

            cluster_count = np.sum(models_data['cluster_labels'] == cluster)
            st.info(f"📈 This style contains **{cluster_count}** artworks in our database")

    with col2:
        st.write("### 💡 Recommended Artworks")

        if uploaded_file is not None:
            with st.spinner("🔍 Finding similar artworks..."):
                recommendations = get_recommendations(features, cluster, top_n=5)

            for rec in recommendations:
                st.markdown("---")
                rec_col1, rec_col2 = st.columns([1, 2])

                with rec_col1:
                    try:
                        rec_image = Image.open(rec['image_path']).convert('RGB')
                        st.image(rec_image, width=150)
                    except:
                        st.warning("⚠️ Image not found")

                with rec_col2:
                    st.markdown(f"**Recommendation #{rec['rank']}**")
                    similarity_pct = rec['similarity'] * 100
                    st.metric("Similarity", f"{similarity_pct:.1f}%")

                    # Progress bar
                    st.progress(rec['similarity'])
        else:
            st.info("👆 Upload an image to get recommendations")

# ============================================================================
# PAGE 2: EXPLORE CLUSTERS
# ============================================================================

elif page == "📊 Explore Clusters":
    st.subheader("Explore Art Style Clusters")

    cluster_names = {
        0: "🖼️ Classical & Portrait",
        1: "🌈 Modern & Abstract",
        2: "🌳 Landscape & Nature"
    }

    # Summary stats
    col1, col2, col3 = st.columns(3)

    for cluster_id in range(3):
        cluster_count = np.sum(models_data['cluster_labels'] == cluster_id)
        percentage = (cluster_count / len(models_data['cluster_labels'])) * 100

        cols = [col1, col2, col3]
        with cols[cluster_id]:
            st.metric(
                get_cluster_name(cluster_id),
                f"{cluster_count:,} artworks",
                f"{percentage:.1f}%"
            )

    st.markdown("---")

    # Show sample images from each cluster
    st.write("### Sample Artworks from Each Style")

    for cluster_id in range(3):
        st.write(f"#### {get_cluster_name(cluster_id)}")

        cluster_mask = models_data['cluster_labels'] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Show 5 random samples
        sample_indices = np.random.choice(cluster_indices, min(5, len(cluster_indices)), replace=False)

        cols = st.columns(5)
        for col_idx, idx in enumerate(sample_indices):
            with cols[col_idx]:
                try:
                    img = Image.open(models_data['image_paths'][idx]).convert('RGB')
                    st.image(img, use_column_width=True)
                except:
                    st.warning("⚠️ Image not found")

# ============================================================================
# PAGE 3: VISUALIZATION
# ============================================================================

elif page == "🔍 Visualization":
    st.subheader("Interactive 2D PCA Visualization")
    st.write("Each point represents an artwork. Points close together have similar styles.")

    # Create interactive plot
    fig = go.Figure()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for cluster_id in range(3):
        mask = models_data['cluster_labels'] == cluster_id

        fig.add_trace(go.Scatter(
            x=models_data['features_2d'][mask, 0],
            y=models_data['features_2d'][mask, 1],
            mode='markers',
            name=get_cluster_name(cluster_id),
            marker=dict(
                size=6,
                color=colors[cluster_id],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=[f"Artwork {i}" for i in range(np.sum(mask))],
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Art Style Clusters (PCA 2D Projection)",
        xaxis_title=f"Principal Component 1 (6.7% variance)",
        yaxis_title=f"Principal Component 2 (5.1% variance)",
        height=600,
        hovermode='closest',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.subheader("About This Project")

    st.markdown("""
    ### 🎨 AI-Powered Art Style Recommender

    This application uses machine learning to analyze artworks and recommend similar pieces based on visual style.

    #### 🔧 Technology Stack
    - **Feature Extraction**: ResNet-50 (pre-trained CNN)
    - **Dimensionality Reduction**: PCA (2D projection for visualization)
    - **Clustering**: K-Means (unsupervised learning)
    - **Recommendation**: Content-based filtering with cosine similarity

    #### 📊 Dataset
    - **10,000 artworks** from WikiArt dataset
    - **3 art styles** automatically detected
    - **2,048-dimensional** feature vectors

    #### 🎯 How It Works
    1. User uploads an artwork image
    2. ResNet-50 extracts 2,048-dimensional features
    3. K-Means predicts which cluster (art style) the image belongs to
    4. System finds similar artworks from the same cluster
    5. Top 5 most similar paintings are recommended

    #### 📈 Clustering Results
    - **Elbow Method**: K=3 optimal clusters
    - **Silhouette Score**: 0.082 (suggesting moderate separation)
    - **Davies-Bouldin Index**: ~2.9 (lower is better)

    ---

    **Project Created**: November 2025

    **Live Demo**: This Streamlit app
    """)

    st.markdown("---")
    st.write("### 📁 Model Files")
    st.json({
        "pca_2d.pkl": "Trained PCA model (2D projection)",
        "kmeans_model.pkl": "Trained K-Means model",
        "cluster_labels.npy": "Cluster assignments for all 10k images",
        "features_2d.npy": "PCA-reduced features (2D)",
        "features.npy": "Full 2048-dim ResNet features",
        "image_paths.pkl": "Paths to artwork images"
    })

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.85rem; margin-top: 2rem;'>
    🎨 Built with Streamlit | ML Pipeline: ResNet → PCA → K-Means<br>
    Dataset: 10,000 artworks | Clusters: 3 art styles
    </div>
""", unsafe_allow_html=True)
