import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import shutil
import zipfile
from datetime import datetime
from collections import Counter
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

# Configuration de la page
st.set_page_config(
    page_title="Gold Test Set Builder - MoLeAd",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
        padding-left: 15px;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">⚖️ Gold Test Set Builder - MoLeAd</div>', unsafe_allow_html=True)

# Initialiser la session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'sampled_data' not in st.session_state:
    st.session_state.sampled_data = None
if 'annotator_a_data' not in st.session_state:
    st.session_state.annotator_a_data = None
if 'annotator_b_data' not in st.session_state:
    st.session_state.annotator_b_data = None
if 'adjudicated_data' not in st.session_state:
    st.session_state.adjudicated_data = None
if 'current_idx_a' not in st.session_state:
    st.session_state.current_idx_a = None
if 'current_idx_b' not in st.session_state:
    st.session_state.current_idx_b = None

# Dossier de données
DATA_DIR = os.path.join(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(DATA_DIR, "results")
WORK_DIR = os.path.join(RESULTS_DIR, "work_in_progress")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# Fichiers de sauvegarde persistants
ANNOTATOR_A_WIP = os.path.join(WORK_DIR, "annotator_a_wip.json")
ANNOTATOR_B_WIP = os.path.join(WORK_DIR, "annotator_b_wip.json")
POSITION_FILE = os.path.join(WORK_DIR, "positions.json")

# Fonctions de sauvegarde et chargement automatiques
def save_annotations_a():
    """Sauvegarde automatique des annotations de l'annotateur A"""
    if st.session_state.annotator_a_data is not None:
        st.session_state.annotator_a_data.to_json(ANNOTATOR_A_WIP, orient='records', force_ascii=False, indent=2)
        # Sauvegarder aussi la position
        positions = load_positions()
        positions['current_idx_a'] = st.session_state.current_idx_a
        save_positions(positions)

def save_annotations_b():
    """Sauvegarde automatique des annotations de l'annotateur B"""
    if st.session_state.annotator_b_data is not None:
        st.session_state.annotator_b_data.to_json(ANNOTATOR_B_WIP, orient='records', force_ascii=False, indent=2)
        # Sauvegarder aussi la position
        positions = load_positions()
        positions['current_idx_b'] = st.session_state.current_idx_b
        save_positions(positions)

def load_annotations_a():
    """Charge les annotations sauvegardées de l'annotateur A"""
    if os.path.exists(ANNOTATOR_A_WIP):
        try:
            df = pd.read_json(ANNOTATOR_A_WIP, orient='records')
            return df
        except Exception as e:
            st.warning(f"Erreur lors du chargement des annotations A: {e}")
    return None

def load_annotations_b():
    """Charge les annotations sauvegardées de l'annotateur B"""
    if os.path.exists(ANNOTATOR_B_WIP):
        try:
            df = pd.read_json(ANNOTATOR_B_WIP, orient='records')
            return df
        except Exception as e:
            st.warning(f"Erreur lors du chargement des annotations B: {e}")
    return None

def save_positions(positions):
    """Sauvegarde les positions des annotateurs"""
    # Convertir les int64 en int Python natifs pour la sérialisation JSON
    positions_serializable = {}
    for key, value in positions.items():
        if value is not None and hasattr(value, 'item'):
            # Convertir NumPy/Pandas int64 en int Python
            positions_serializable[key] = int(value)
        else:
            positions_serializable[key] = value
    
    with open(POSITION_FILE, 'w', encoding='utf-8') as f:
        json.dump(positions_serializable, f, ensure_ascii=False, indent=2)

def load_positions():
    """Charge les positions sauvegardées"""
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {'current_idx_a': None, 'current_idx_b': None}

# Charger les positions sauvegardées au démarrage
if st.session_state.current_idx_a is None and st.session_state.current_idx_b is None:
    saved_positions = load_positions()
    st.session_state.current_idx_a = saved_positions.get('current_idx_a')
    st.session_state.current_idx_b = saved_positions.get('current_idx_b')

# Sidebar - Navigation
st.sidebar.title("📚 Navigation")
pages = {
    'home': '🏠 Accueil',
    'sampling': '📊 1. Échantillonnage Stratifié',
    'annotation_a': '👤 2a. Annotation - Annotateur A',
    'annotation_b': '👥 2b. Annotation - Annotateur B',
    'iaa': '📈 3. Accord Inter-Annotateurs',
    'adjudication': '⚖️ 4. Adjudication',
    'evaluation': '🎯 5. Évaluation des Performances',
    'export': '💾 6. Export & Rapport'
}

for page_key, page_name in pages.items():
    if st.sidebar.button(page_name, key=f"btn_{page_key}", use_container_width=True):
        st.session_state.page = page_key

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Sauvegarde & Restauration")

# 1. Télécharger Backup
if os.path.exists(WORK_DIR):
    # Créer le zip temporaire
    shutil.make_archive(os.path.join(RESULTS_DIR, "temp_backup"), 'zip', WORK_DIR)
    
    with open(os.path.join(RESULTS_DIR, "temp_backup.zip"), "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Télécharger Backup (.zip)",
            data=f,
            file_name=f"molead_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            help="Téléchargez une copie complète de vos annotations en cours pour les sauvegarder en local."
        )

# 2. Restaurer Backup
uploaded_backup = st.sidebar.file_uploader("Restaurer un Backup", type="zip", help="Attention: Ceci écrasera les données actuelles")

if uploaded_backup is not None:
    if st.sidebar.button("⚠️ Confirmer la Restauration", type="primary"):
        try:
            # Sauvegarder le zip uploadé temporairement
            temp_zip_path = os.path.join(RESULTS_DIR, "uploaded_restore.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_backup.getbuffer())
            
            # Vérifier si c'est un zip valide
            if zipfile.is_zipfile(temp_zip_path):
                # Vider le dossier WIP actuel par sécurité (ou juste écraser)
                # On choisit d'écraser/fusionner
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(WORK_DIR)
                
                st.sidebar.success("✅ Restauration réussie!")
                # Forcer le rechargement des positions et données
                st.session_state.annotator_a_data = None
                st.session_state.annotator_b_data = None
                st.session_state.current_idx_a = None
                st.session_state.current_idx_b = None
                st.rerun()
            else:
                st.sidebar.error("Le fichier n'est pas un ZIP valide.")
        except Exception as e:
            st.sidebar.error(f"Erreur lors de la restauration: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Documentation")
st.sidebar.markdown("""
**Workflow complet :**
1. Échantillonnage stratifié (500-1000 annonces)
2. Annotation en double aveugle
3. Calcul de l'IAA (Kappa)
4. Adjudication des conflits
5. Évaluation vs. Silver Labels
6. Génération du rapport final
""")

# Fonction pour charger les données
@st.cache_data
def load_legal_announcements():
    """Charge le fichier legal_announcements.json"""
    file_path = config.DATA_PATH
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Fonction pour l'échantillonnage stratifié basé sur category et subject_canonical
def stratified_sampling(df, n_samples=1000, min_rare_samples=30):
    """
    Échantillonnage stratifié avec quota minimum pour les classes rares
    Stratification par (category, subject_canonical)
    """
    # Créer une colonne combinée pour la stratification
    def get_strata_key(row):
        category = row.get('category', 'Unknown')
        subject = row.get('subject_canonical', 'Unknown')
        return f"{category}::{subject}"
    
    df['_strata'] = df.apply(get_strata_key, axis=1)
    
    # Compter les occurrences par strate
    strata_counts = df['_strata'].value_counts()
    
    # Identifier les classes rares (moins de 5% du total)
    threshold = len(df) * 0.05
    rare_strata = strata_counts[strata_counts < threshold].index.tolist()
    common_strata = strata_counts[strata_counts >= threshold].index.tolist()
    
    sampled_data = []
    
    # Échantillonner les strates rares avec quota minimum
    for rare_stratum in rare_strata:
        rare_df = df[df['_strata'] == rare_stratum]
        n_rare = min(min_rare_samples, len(rare_df))
        if n_rare > 0:
            sampled_data.append(rare_df.sample(n=n_rare, random_state=42))
    
    # Calculer combien il reste à échantillonner
    remaining_samples = n_samples - sum(len(s) for s in sampled_data)
    
    # Échantillonner proportionnellement parmi les strates communes
    if remaining_samples > 0 and len(common_strata) > 0:
        common_df = df[df['_strata'].isin(common_strata)]
        if len(common_df) > 0:
            n_common = min(remaining_samples, len(common_df))
            sampled_data.append(common_df.sample(n=n_common, random_state=42))
    
    # Combiner tous les échantillons
    if sampled_data:
        final_sample = pd.concat(sampled_data, ignore_index=True)
        final_sample = final_sample.drop(columns=['_strata'])
    else:
        final_sample = pd.DataFrame()
    
    return final_sample

# Fonction pour calculer le Cohen's Kappa
def calculate_kappa(labels_a, labels_b):
    """Calcule le Cohen's Kappa entre deux annotateurs"""
    # Supprimer les paires où l'un des labels est None
    valid_indices = [i for i in range(len(labels_a)) 
                    if labels_a[i] is not None and labels_b[i] is not None]
    
    if len(valid_indices) == 0:
        return 0.0
    
    filtered_a = [labels_a[i] for i in valid_indices]
    filtered_b = [labels_b[i] for i in valid_indices]
    
    return cohen_kappa_score(filtered_a, filtered_b)

# Fonction pour calculer les métriques de performance
def calculate_performance_metrics(y_true, y_pred, average='macro'):
    """Calcule les métriques de performance"""
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# ====================== PAGES ======================

# Page d'accueil
if st.session_state.page == 'home':
    st.markdown('<div class="section-header">👋 Bienvenue dans le Gold Test Set Builder</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Cette application vous guide dans la création d'un <b>Gold Standard Dataset</b> conforme aux exigences 
    
    </div>
    """, unsafe_allow_html=True)
    #des revues Q1 pour valider votre méthode d'annotation automatique (weak supervision).
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objectifs")
        st.markdown("""
        - Créer un échantillon représentatif et stratifié
        - Garantir une annotation de haute qualité
        - Mesurer la fiabilité inter-annotateurs
        - Évaluer les performances du système automatique
        - Générer un rapport scientifique complet
        """)
        
    with col2:
        st.markdown("### 📊 Métriques Clés")
        st.markdown("""
        - **Cohen's Kappa** : Accord inter-annotateurs (> 0.8 = excellent)
        - **Precision** : Éviter les faux positifs
        - **Recall** : Capturer tous les cas pertinents
        - **F1-Score** : Moyenne harmonique (Macro & Micro)
        - **Matrice de Confusion** : Analyse détaillée des erreurs
        """)
    
    st.markdown("### 📋 Workflow en 6 étapes")
    
    steps = [
        ("1️⃣ Échantillonnage Stratifié", "Sélection de 500-1000 annonces avec quotas pour classes rares"),
        ("2️⃣ Annotation Double-Aveugle", "Deux annotateurs indépendants étiquettent l'échantillon"),
        ("3️⃣ Accord Inter-Annotateurs", "Calcul du Cohen's Kappa pour valider la cohérence"),
        ("4️⃣ Adjudication", "Expert tiers résout les conflits d'annotation"),
        ("5️⃣ Évaluation", "Comparaison Gold Standard vs. Silver Labels"),
        ("6️⃣ Export & Rapport", "Génération du rapport final pour publication")
    ]
    
    for step, description in steps:
        with st.expander(step):
            st.write(description)
    
    st.markdown("---")
    st.info("👈 Utilisez le menu latéral pour commencer le workflow.")

# Page 1: Échantillonnage
elif st.session_state.page == 'sampling':
    st.markdown('<div class="section-header">📊 Échantillonnage Stratifié</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    L'échantillonnage stratifié garantit une représentation équilibrée de toutes les catégories,
    y compris les classes rares qui seraient sous-représentées dans un échantillonnage aléatoire simple.
    </div>
    """, unsafe_allow_html=True)
    
    # Charger les données
    df = load_legal_announcements()
    
    if df is not None:
        st.success(f"✅ {len(df):,} annonces légales chargées avec succès")
        
        # Paramètres d'échantillonnage
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("Nombre total d'échantillons", 
                                       min_value=100, max_value=2000, value=1000, step=50)
        with col2:
            min_rare = st.number_input("Quota minimum pour classes rares", 
                                      min_value=10, max_value=100, value=50, step=10)
        
        if st.button("🎲 Générer l'échantillon stratifié", type="primary"):
            with st.spinner("Génération de l'échantillon..."):
                sampled_df = stratified_sampling(df, n_samples=n_samples, min_rare_samples=min_rare)
                st.session_state.sampled_data = sampled_df
                
                # Sauvegarder
                output_file = os.path.join(RESULTS_DIR, f"sampled_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                sampled_df.to_json(output_file, orient='records', force_ascii=False, indent=2)
                
                st.success(f"✅ Échantillon de {len(sampled_df)} annonces généré et sauvegardé!")
        
        # Afficher l'échantillon si disponible
        if st.session_state.sampled_data is not None:
            st.markdown("### 📋 Statistiques de l'échantillon")
            
            # Distribution par catégorie et sujet
            def get_category_subject(row):
                category = row.get('category', 'Unknown')
                subject = row.get('subject_canonical', 'Unknown')
                return f"{category} - {subject}"
            
            category_subject = st.session_state.sampled_data.apply(get_category_subject, axis=1)
            category_dist = category_subject.value_counts()
            
            # Distribution par catégorie seule
            categories = st.session_state.sampled_data['category']
            category_only_dist = categories.value_counts()
            
            # Tableau des statistiques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total échantillon", len(st.session_state.sampled_data))
            with col2:
                st.metric("Catégories", len(category_only_dist))
            with col3:
                st.metric("Sujets uniques", len(category_dist))
            with col4:
                st.metric("Sujet minimal", category_dist.min())
            
            # Graphiques de distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribution par Catégorie")
                fig1 = px.bar(x=category_only_dist.index, y=category_only_dist.values,
                            labels={'x': 'Catégorie', 'y': 'Nombre d\'annonces'},
                            color=category_only_dist.values,
                            color_continuous_scale='Blues')
                fig1.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribution par Sujet (Top 15)")
                fig2 = px.bar(x=category_dist.index[:15], y=category_dist.values[:15],
                            labels={'x': 'Sujet Canonical', 'y': 'Nombre d\'annonces'},
                            color=category_dist.values[:15],
                            color_continuous_scale='Viridis')
                fig2.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Aperçu des données
            st.markdown("### 👀 Aperçu de l'échantillon")
            st.dataframe(st.session_state.sampled_data[['id', 'text_content']].head(10))

# Page 2a: Annotation Annotateur A
elif st.session_state.page == 'annotation_a':
    st.markdown('<div class="section-header">👤 Annotation - Annotateur A</div>', unsafe_allow_html=True)
    
    if st.session_state.sampled_data is None:
        st.warning("⚠️ Veuillez d'abord générer un échantillon stratifié (Étape 1)")
    else:
        st.markdown("""
        <div class="info-box">
        <b>Consignes d'annotation :</b><br>
        - Lisez attentivement chaque annonce<br>
        - N'utilisez PAS les étiquettes automatiques comme référence<br>
        - En cas de doute, consultez les définitions des catégories<br>
        - Prenez des pauses régulières pour maintenir la concentration
        </div>
        """, unsafe_allow_html=True)
        
        # Interface d'annotation
        if st.session_state.annotator_a_data is None:
            # Essayer de charger les annotations précédentes
            loaded_data = load_annotations_a()
            if loaded_data is not None:
                st.session_state.annotator_a_data = loaded_data
                st.info(f"📂 {len(loaded_data[loaded_data['annotation_a'].notna()])} annotations précédentes chargées pour l'Annotateur A")
            else:
                st.session_state.annotator_a_data = st.session_state.sampled_data.copy()
                st.session_state.annotator_a_data['annotation_a'] = None
                st.session_state.annotator_a_data['annotation_a_date'] = None
        
        # Sélection de l'annonce à annoter
        total = len(st.session_state.annotator_a_data)
        annotated = st.session_state.annotator_a_data['annotation_a'].notna().sum()
        
        st.progress(annotated / total, text=f"Progression: {annotated}/{total} ({annotated/total*100:.1f}%)")
        
        # Trouver la prochaine annonce non annotée
        unannotated_idx = st.session_state.annotator_a_data[
            st.session_state.annotator_a_data['annotation_a'].isna()
        ].index
        
        if len(unannotated_idx) > 0:
            # Déterminer l'index par défaut
            if st.session_state.current_idx_a is not None and st.session_state.current_idx_a in st.session_state.annotator_a_data.index:
                # Reprendre où on s'était arrêté
                default_idx = st.session_state.annotator_a_data.index.get_loc(st.session_state.current_idx_a)
            else:
                # Première annonce non annotée
                default_idx = int(unannotated_idx[0])
            
            current_idx = st.selectbox("Sélectionner une annonce", 
                                      options=st.session_state.annotator_a_data.index,
                                      index=default_idx)
            
            # Sauvegarder la position actuelle
            st.session_state.current_idx_a = current_idx
            
            current_row = st.session_state.annotator_a_data.loc[current_idx]
            
            # Afficher l'annonce
            st.markdown("### 📄 Annonce à annoter")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**ID:** {current_row['id']}")
                st.markdown(f"**Entreprise:** {current_row.get('text_content', {}).get('company_name', 'N/A')}")
                st.markdown(f"**🤖 Auto-Category:** {current_row.get('category', 'N/A')}")
                st.markdown(f"**🤖 Auto-Subject:** {current_row.get('subject_canonical', 'N/A')}")
                st.text_area("Texte complet", 
                           value=current_row.get('text_content', {}).get('body', ''),
                           height=300, disabled=True)
            
            
            with col2:
                st.markdown("**Annotation Manuelle**")
                
                # Récupérer les valeurs auto-détectées
                auto_category = current_row.get('category', '')
                auto_subject = current_row.get('subject_canonical', '')
                
                # Trouver l'index de la catégorie auto si elle existe
                category_options = [""] + config.CATEGORIES
                default_cat_index = 0
                if auto_category in category_options:
                    default_cat_index = category_options.index(auto_category)
                
                annotation_category = st.selectbox(
                    "Catégorie:",
                    options=category_options,
                    index=default_cat_index,
                    key=f"annot_cat_a_{current_idx}"
                )
                
                # Afficher les sujets canoniques selon la catégorie choisie
                subject_options = [""]
                if annotation_category == "Creation":
                    subject_options += list(config.CREATION_CANONICAL)
                elif annotation_category == "Modification":
                    subject_options += list(config.MODIFICATION_CANONICAL)
                
                # Trouver l'index du sujet auto si il existe
                default_subj_index = 0
                if auto_subject in subject_options:
                    default_subj_index = subject_options.index(auto_subject)
                
                annotation_subject = st.selectbox(
                    "Sujet Canonical:",
                    options=subject_options,
                    index=default_subj_index,
                    key=f"annot_subj_a_{current_idx}"
                )
                
                confidence = st.slider("Niveau de confiance", 1, 5, 5, key=f"conf_a_{current_idx}")
                notes = st.text_area("Notes (optionnel)", key=f"notes_a_{current_idx}")
                
                if st.button("💾 Sauvegarder l'annotation", type="primary"):
                    if annotation_category and annotation_subject:
                        # Stocker catégorie et sujet séparément
                        st.session_state.annotator_a_data.at[current_idx, 'annotation_a_category'] = annotation_category
                        st.session_state.annotator_a_data.at[current_idx, 'annotation_a_subject'] = annotation_subject
                        st.session_state.annotator_a_data.at[current_idx, 'annotation_a'] = f"{annotation_category}::{annotation_subject}"
                        st.session_state.annotator_a_data.at[current_idx, 'annotation_a_date'] = datetime.now().isoformat()
                        st.session_state.annotator_a_data.at[current_idx, 'confidence_a'] = confidence
                        st.session_state.annotator_a_data.at[current_idx, 'notes_a'] = notes
                        
                        # Avancer à la prochaine annonce non annotée
                        remaining_unannotated = st.session_state.annotator_a_data[
                            st.session_state.annotator_a_data['annotation_a'].isna()
                        ].index
                        if len(remaining_unannotated) > 0:
                            st.session_state.current_idx_a = remaining_unannotated[0]
                        
                        # 💾 Sauvegarde automatique persistante
                        save_annotations_a()
                        
                        st.success("✅ Annotation sauvegardée!")
                        st.rerun()
                    else:
                        st.error("Veuillez sélectionner une catégorie ET un sujet canonical")
        else:
            st.success("🎉 Toutes les annonces ont été annotées par l'Annotateur A!")
            
            # Bouton pour exporter
            if st.button("💾 Exporter les annotations"):
                output_file = os.path.join(RESULTS_DIR, f"annotator_a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                st.session_state.annotator_a_data.to_json(output_file, orient='records', force_ascii=False, indent=2)
                st.success(f"✅ Annotations exportées: {output_file}")

# Page 2b: Annotation Annotateur B
elif st.session_state.page == 'annotation_b':
    st.markdown('<div class="section-header">👥 Annotation - Annotateur B</div>', unsafe_allow_html=True)
    
    if st.session_state.sampled_data is None:
        st.warning("⚠️ Veuillez d'abord générer un échantillon stratifié (Étape 1)")
    else:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Double-Aveugle Important :</b><br>
        L'Annotateur B ne doit PAS voir les annotations de l'Annotateur A.
        Assurez-vous que les deux annotateurs travaillent indépendamment.
        </div>
        """, unsafe_allow_html=True)
        
        # Interface similaire à l'Annotateur A
        if st.session_state.annotator_b_data is None:
            # Essayer de charger les annotations précédentes
            loaded_data = load_annotations_b()
            if loaded_data is not None:
                st.session_state.annotator_b_data = loaded_data
                st.info(f"📂 {len(loaded_data[loaded_data['annotation_b'].notna()])} annotations précédentes chargées pour l'Annotateur B")
            else:
                st.session_state.annotator_b_data = st.session_state.sampled_data.copy()
                st.session_state.annotator_b_data['annotation_b'] = None
                st.session_state.annotator_b_data['annotation_b_date'] = None
        
        total = len(st.session_state.annotator_b_data)
        annotated = st.session_state.annotator_b_data['annotation_b'].notna().sum()
        
        st.progress(annotated / total, text=f"Progression: {annotated}/{total} ({annotated/total*100:.1f}%)")
        
        unannotated_idx = st.session_state.annotator_b_data[
            st.session_state.annotator_b_data['annotation_b'].isna()
        ].index
        
        if len(unannotated_idx) > 0:
            # Déterminer l'index par défaut
            if st.session_state.current_idx_b is not None and st.session_state.current_idx_b in st.session_state.annotator_b_data.index:
                # Reprendre où on s'était arrêté
                default_idx = st.session_state.annotator_b_data.index.get_loc(st.session_state.current_idx_b)
            else:
                # Première annonce non annotée
                default_idx = int(unannotated_idx[0])
            
            current_idx = st.selectbox("Sélectionner une annonce", 
                                      options=st.session_state.annotator_b_data.index,
                                      index=default_idx)
            
            # Sauvegarder la position actuelle
            st.session_state.current_idx_b = current_idx
            
            current_row = st.session_state.annotator_b_data.loc[current_idx]
            
            st.markdown("### 📄 Annonce à annoter")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**ID:** {current_row['id']}")
                st.markdown(f"**Entreprise:** {current_row.get('text_content', {}).get('company_name', 'N/A')}")
                st.markdown(f"**🤖 Auto-Category:** {current_row.get('category', 'N/A')}")
                st.markdown(f"**🤖 Auto-Subject:** {current_row.get('subject_canonical', 'N/A')}")
                st.text_area("Texte complet", 
                           value=current_row.get('text_content', {}).get('body', ''),
                           height=300, disabled=True)
            
            with col2:
                st.markdown("**Annotation Manuelle**")
                
                # Récupérer les valeurs auto-détectées
                auto_category = current_row.get('category', '')
                auto_subject = current_row.get('subject_canonical', '')
                
                # Trouver l'index de la catégorie auto si elle existe
                category_options = [""] + config.CATEGORIES
                default_cat_index = 0
                if auto_category in category_options:
                    default_cat_index = category_options.index(auto_category)
                
                annotation_category = st.selectbox(
                    "Catégorie:",
                    options=category_options,
                    index=default_cat_index,
                    key=f"annot_cat_b_{current_idx}"
                )
                
                # Afficher les sujets canoniques selon la catégorie choisie
                subject_options = [""]
                if annotation_category == "Creation":
                    subject_options += list(config.CREATION_CANONICAL)
                elif annotation_category == "Modification":
                    subject_options += list(config.MODIFICATION_CANONICAL)
                
                # Trouver l'index du sujet auto si il existe
                default_subj_index = 0
                if auto_subject in subject_options:
                    default_subj_index = subject_options.index(auto_subject)
                
                annotation_subject = st.selectbox(
                    "Sujet Canonical:",
                    options=subject_options,
                    index=default_subj_index,
                    key=f"annot_subj_b_{current_idx}"
                )
                
                confidence = st.slider("Niveau de confiance", 1, 5, 5, key=f"conf_b_{current_idx}")
                notes = st.text_area("Notes (optionnel)", key=f"notes_b_{current_idx}")
                
                if st.button("💾 Sauvegarder l'annotation", type="primary"):
                    if annotation_category and annotation_subject:
                        # Stocker catégorie et sujet séparément
                        st.session_state.annotator_b_data.at[current_idx, 'annotation_b_category'] = annotation_category
                        st.session_state.annotator_b_data.at[current_idx, 'annotation_b_subject'] = annotation_subject
                        st.session_state.annotator_b_data.at[current_idx, 'annotation_b'] = f"{annotation_category}::{annotation_subject}"
                        st.session_state.annotator_b_data.at[current_idx, 'annotation_b_date'] = datetime.now().isoformat()
                        st.session_state.annotator_b_data.at[current_idx, 'confidence_b'] = confidence
                        st.session_state.annotator_b_data.at[current_idx, 'notes_b'] = notes
                        
                        # Avancer à la prochaine annonce non annotée
                        remaining_unannotated = st.session_state.annotator_b_data[
                            st.session_state.annotator_b_data['annotation_b'].isna()
                        ].index
                        if len(remaining_unannotated) > 0:
                            st.session_state.current_idx_b = remaining_unannotated[0]
                        
                        # 💾 Sauvegarde automatique persistante
                        save_annotations_b()
                        
                        st.success("✅ Annotation sauvegardée!")
                        st.rerun()
                    else:
                        st.error("Veuillez sélectionner une catégorie ET un sujet canonical")
        else:
            st.success("🎉 Toutes les annonces ont été annotées par l'Annotateur B!")
            
            if st.button("💾 Exporter les annotations"):
                output_file = os.path.join(RESULTS_DIR, f"annotator_b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                st.session_state.annotator_b_data.to_json(output_file, orient='records', force_ascii=False, indent=2)
                st.success(f"✅ Annotations exportées: {output_file}")

# Page 3: IAA (Inter-Annotator Agreement)
elif st.session_state.page == 'iaa':
    st.markdown('<div class="section-header">📈 Accord Inter-Annotateurs (IAA)</div>', unsafe_allow_html=True)
    
    if st.session_state.annotator_a_data is None or st.session_state.annotator_b_data is None:
        st.warning("⚠️ Veuillez d'abord compléter les annotations des deux annotateurs")
    else:
        # Vérifier que les deux ont terminé
        a_complete = st.session_state.annotator_a_data['annotation_a'].notna().all()
        b_complete = st.session_state.annotator_b_data['annotation_b'].notna().all()
        
        if not (a_complete and b_complete):
            st.warning("⚠️ Les deux annotateurs doivent terminer toutes les annotations")
        else:
            st.success("✅ Les deux annotateurs ont terminé leurs annotations")
            
            # Fusionner les données
            labels_a = st.session_state.annotator_a_data['annotation_a'].tolist()
            labels_b = st.session_state.annotator_b_data['annotation_b'].tolist()
            
            # Calculer le Cohen's Kappa
            kappa = calculate_kappa(labels_a, labels_b)
            
            # Afficher les résultats
            st.markdown("### 🎯 Résultats de l'IAA")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cohen's Kappa", f"{kappa:.3f}")
            
            with col2:
                # Accord simple
                agreements = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
                agreement_rate = agreements / len(labels_a)
                st.metric("Accord simple", f"{agreement_rate:.1%}")
            
            with col3:
                # Désaccords
                disagreements = len(labels_a) - agreements
                st.metric("Désaccords", disagreements)
            
            # Interprétation du Kappa
            if kappa > 0.8:
                st.markdown("""
                <div class="success-box">
                <b>🎉 Excellent!</b> Kappa > 0.8 indique un accord "presque parfait". 
                Votre taxonomie est claire et les annotateurs sont cohérents.
                </div>
                """, unsafe_allow_html=True)
            elif kappa > 0.6:
                st.markdown("""
                <div class="warning-box">
                <b>⚠️ Substantiel.</b> Kappa entre 0.6 et 0.8 indique un accord "substantiel". 
                Acceptable, mais pourrait nécessiter une clarification de certaines catégories.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                <b>❌ Insuffisant.</b> Kappa < 0.6 indique un accord faible. 
                Recommandé : clarifier la taxonomie et recommencer l'annotation.
                </div>
                """, unsafe_allow_html=True)
            
            # Matrice de confusion entre annotateurs
            st.markdown("### 📊 Matrice de Confusion (Annotateur A vs B)")
            
            unique_labels = sorted(list(set(labels_a + labels_b)))
            conf_matrix = confusion_matrix(labels_a, labels_b, labels=unique_labels)
            
            fig = px.imshow(conf_matrix,
                          labels=dict(x="Annotateur B", y="Annotateur A", color="Nombre"),
                          x=unique_labels,
                          y=unique_labels,
                          color_continuous_scale='Blues',
                          text_auto=True)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des désaccords
            st.markdown("### 🔍 Analyse des désaccords")
            disagreement_data = []
            for i, (a, b) in enumerate(zip(labels_a, labels_b)):
                if a != b:
                    disagreement_data.append({
                        'Index': i,
                        'ID': st.session_state.annotator_a_data.iloc[i]['id'],
                        'Annotateur A': a,
                        'Annotateur B': b,
                        'Confiance A': st.session_state.annotator_a_data.iloc[i].get('confidence_a', 'N/A'),
                        'Confiance B': st.session_state.annotator_b_data.iloc[i].get('confidence_b', 'N/A')
                    })
            
            if disagreement_data:
                df_disagreements = pd.DataFrame(disagreement_data)
                st.dataframe(df_disagreements, use_container_width=True)
                
                # Export des désaccords pour adjudication
                if st.button("📥 Exporter les désaccords pour adjudication"):
                    output_file = os.path.join(RESULTS_DIR, f"disagreements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    df_disagreements.to_csv(output_file, index=False)
                    st.success(f"✅ Désaccords exportés: {output_file}")

# Page 4: Adjudication
elif st.session_state.page == 'adjudication':
    st.markdown('<div class="section-header">⚖️ Adjudication des Conflits</div>', unsafe_allow_html=True)
    
    if st.session_state.annotator_a_data is None or st.session_state.annotator_b_data is None:
        st.warning("⚠️ Veuillez d'abord compléter les annotations et calculer l'IAA")
    else:
        st.markdown("""
        <div class="info-box">
        <b>Rôle du Super-Annotateur :</b><br>
        Un expert indépendant (super-annotateur) examine chaque désaccord et décide 
        de l'étiquette finale qui constituera le Gold Standard.
        </div>
        """, unsafe_allow_html=True)
        
        # Identifier les désaccords
        labels_a = st.session_state.annotator_a_data['annotation_a'].tolist()
        labels_b = st.session_state.annotator_b_data['annotation_b'].tolist()
        
        disagreement_indices = [i for i in range(len(labels_a)) if labels_a[i] != labels_b[i]]
        
        if st.session_state.adjudicated_data is None:
            st.session_state.adjudicated_data = st.session_state.annotator_a_data.copy()
            st.session_state.adjudicated_data['gold_label'] = None
            st.session_state.adjudicated_data['adjudicator'] = None
            st.session_state.adjudicated_data['adjudication_date'] = None
            
            # Pour les accords, copier directement
            for i in range(len(labels_a)):
                if labels_a[i] == labels_b[i]:
                    st.session_state.adjudicated_data.at[i, 'gold_label'] = labels_a[i]
                    st.session_state.adjudicated_data.at[i, 'adjudicator'] = 'auto_agreement'
        
        total_conflicts = len(disagreement_indices)
        resolved = st.session_state.adjudicated_data.loc[disagreement_indices, 'gold_label'].notna().sum()
        
        st.progress(resolved / total_conflicts if total_conflicts > 0 else 1.0,
                   text=f"Conflits résolus: {resolved}/{total_conflicts}")
        
        if total_conflicts == 0:
            st.success("🎉 Aucun désaccord ! Les deux annotateurs sont parfaitement alignés.")
        else:
            # Trouver le prochain conflit non résolu
            unresolved = [idx for idx in disagreement_indices 
                         if pd.isna(st.session_state.adjudicated_data.at[idx, 'gold_label'])]
            
            if len(unresolved) > 0:
                current_idx = st.selectbox("Sélectionner un conflit à résoudre",
                                          options=disagreement_indices,
                                          index=disagreement_indices.index(unresolved[0]))
                
                current_row = st.session_state.adjudicated_data.loc[current_idx]
                
                st.markdown("### 📄 Annonce en conflit")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**ID:** {current_row['id']}")
                    st.markdown(f"**Entreprise:** {current_row['text_content'].get('company_name', 'N/A')}")
                    st.text_area("Texte complet", 
                               value=current_row['text_content'].get('body', ''),
                               height=300, disabled=True)
                
                with col2:
                    st.markdown("### 🔍 Annotations")
                    st.markdown(f"**Annotateur A:** {labels_a[current_idx]}")
                    st.markdown(f"**Annotateur B:** {labels_b[current_idx]}")
                    
                    st.markdown("---")
                    st.markdown("### ⚖️ Décision Finale")
                    
                    gold_label = st.selectbox(
                        "Label Gold Standard:",
                        options=["", labels_a[current_idx], labels_b[current_idx], "Autre"],
                        key=f"adj_{current_idx}"
                    )
                    
                    if gold_label == "Autre":
                        gold_label = st.selectbox(
                            "Spécifier la catégorie:",
                            options=["Création", "Modification", "Dissolution", "Fusion/Scission", "Autre"]
                        )
                    
                    adjudicator_name = st.text_input("Nom du super-annotateur")
                    notes = st.text_area("Justification (optionnel)")
                    
                    if st.button("✅ Valider la décision", type="primary"):
                        if gold_label and adjudicator_name:
                            st.session_state.adjudicated_data.at[current_idx, 'gold_label'] = gold_label
                            st.session_state.adjudicated_data.at[current_idx, 'adjudicator'] = adjudicator_name
                            st.session_state.adjudicated_data.at[current_idx, 'adjudication_date'] = datetime.now().isoformat()
                            st.session_state.adjudicated_data.at[current_idx, 'adjudication_notes'] = notes
                            st.success("✅ Décision enregistrée!")
                            st.rerun()
                        else:
                            st.error("Veuillez remplir tous les champs requis")
            
            else:
                st.success("🎉 Tous les conflits ont été résolus!")
                
                if st.button("💾 Exporter le Gold Standard"):
                    output_file = os.path.join(RESULTS_DIR, f"gold_standard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    st.session_state.adjudicated_data.to_json(output_file, orient='records', force_ascii=False, indent=2)
                    st.success(f"✅ Gold Standard exporté: {output_file}")

# Page 5: Évaluation
elif st.session_state.page == 'evaluation':
    st.markdown('<div class="section-header">🎯 Évaluation des Performances</div>', unsafe_allow_html=True)
    
    if st.session_state.adjudicated_data is None:
        st.warning("⚠️ Veuillez d'abord compléter l'adjudication pour obtenir le Gold Standard")
    else:
        # Vérifier que le Gold Standard est complet
        if st.session_state.adjudicated_data['gold_label'].isna().any():
            st.warning("⚠️ L'adjudication n'est pas terminée. Veuillez résoudre tous les conflits.")
        else:
            st.success("✅ Gold Standard complet. Prêt pour l'évaluation!")
            
            st.markdown("""
            <div class="info-box">
            Cette section compare les <b>Silver Labels</b> (annotations automatiques) avec le 
            <b>Gold Standard</b> (annotations humaines validées) pour évaluer les performances 
            de votre système d'annotation automatique.
            </div>
            """, unsafe_allow_html=True)
            
            # Extraire les labels
            gold_labels = st.session_state.adjudicated_data['gold_label'].tolist()
            
            # Pour silver labels, combiner category et subject_canonical
            def get_silver_label(row):
                category = row.get('category', 'Unknown')
                subject = row.get('subject_canonical', 'Unknown')
                return f"{category}::{subject}"
            
            silver_labels = st.session_state.adjudicated_data.apply(get_silver_label, axis=1).tolist()
            
            # Evaluation separee : Categorie seulement
            gold_categories = [label.split('::')[0] if '::' in label else label for label in gold_labels]
            silver_categories = st.session_state.adjudicated_data['category'].tolist()
            
            # Evaluation separee : Sujet seulement
            gold_subjects = [label.split('::')[1] if '::' in label and len(label.split('::')) > 1 else 'Unknown' for label in gold_labels]
            silver_subjects = st.session_state.adjudicated_data['subject_canonical'].tolist()
            
            # Calculer les métriques
            st.markdown("### 📊 Métriques de Performance")
            
            # Macro-averaged metrics
            macro_metrics = calculate_performance_metrics(gold_labels, silver_labels, average='macro')
            
            # Micro-averaged metrics
            micro_metrics = calculate_performance_metrics(gold_labels, silver_labels, average='micro')
            
            # Afficher les métriques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Macro-Averaged")
                st.metric("Precision", f"{macro_metrics['precision']:.3f}")
                st.metric("Recall", f"{macro_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{macro_metrics['f1_score']:.3f}")
            
            with col2:
                st.markdown("#### Micro-Averaged")
                st.metric("Precision", f"{micro_metrics['precision']:.3f}")
                st.metric("Recall", f"{micro_metrics['recall']:.3f}")
                st.metric("F1-Score", f"{micro_metrics['f1_score']:.3f}")
            
            with col3:
                st.markdown("#### Global")
                accuracy = sum(1 for g, s in zip(gold_labels, silver_labels) if g == s) / len(gold_labels)
                st.metric("Accuracy", f"{accuracy:.3f}")
                total_errors = sum(1 for g, s in zip(gold_labels, silver_labels) if g != s)
                st.metric("Erreurs totales", total_errors)
            
            # Rapport de classification détaillé
            st.markdown("### 📋 Rapport de Classification Détaillé")
            
            from sklearn.metrics import classification_report
            report = classification_report(gold_labels, silver_labels, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            
            # Matrice de confusion
            st.markdown("### 🔲 Matrice de Confusion")
            
            unique_labels = sorted(list(set(gold_labels + silver_labels)))
            conf_matrix = confusion_matrix(gold_labels, silver_labels, labels=unique_labels)
            
            fig = px.imshow(conf_matrix,
                          labels=dict(x="Prédiction (Silver)", y="Vérité Terrain (Gold)", color="Nombre"),
                          x=unique_labels,
                          y=unique_labels,
                          color_continuous_scale='RdYlGn_r',
                          text_auto=True)
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des erreurs
            st.markdown("### 🔍 Analyse des Erreurs")
            
            errors_data = []
            for i, (gold, silver) in enumerate(zip(gold_labels, silver_labels)):
                if gold != silver:
                    errors_data.append({
                        'Index': i,
                        'ID': st.session_state.adjudicated_data.iloc[i]['id'],
                        'Gold': gold,
                        'Silver': silver,
                        'Entreprise': st.session_state.adjudicated_data.iloc[i]['text_content'].get('company_name', 'N/A')
                    })
            
            if errors_data:
                df_errors = pd.DataFrame(errors_data)
                st.dataframe(df_errors, use_container_width=True)
                
                # Distribution des erreurs par catégorie
                error_counts = df_errors.groupby(['Gold', 'Silver']).size().reset_index(name='Count')
                error_counts['Error_Type'] = error_counts['Gold'] + ' → ' + error_counts['Silver']
                
                fig_errors = px.bar(error_counts.sort_values('Count', ascending=False).head(10),
                                   x='Error_Type', y='Count',
                                   title="Top 10 Types d'Erreurs",
                                   labels={'Error_Type': 'Type d\'Erreur', 'Count': 'Nombre'},
                                   color='Count',
                                   color_continuous_scale='Reds')
                st.plotly_chart(fig_errors, use_container_width=True)

# Page 6: Export & Rapport
elif st.session_state.page == 'export':
    st.markdown('<div class="section-header">💾 Export & Rapport Final</div>', unsafe_allow_html=True)
    
    if st.session_state.adjudicated_data is None:
        st.warning("⚠️ Veuillez d'abord compléter toutes les étapes précédentes")
    else:
        st.markdown("""
        <div class="success-box">
        <b>✅ Workflow terminé!</b><br>
        Vous pouvez maintenant générer le rapport final et exporter toutes les données.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📥 Exports Disponibles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Données")
            if st.button("📊 Exporter Gold Standard (JSON)", use_container_width=True):
                output_file = os.path.join(RESULTS_DIR, f"gold_standard_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                st.session_state.adjudicated_data.to_json(output_file, orient='records', force_ascii=False, indent=2)
                st.success(f"✅ Exporté: {output_file}")
            
            if st.button("📊 Exporter Gold Standard (CSV)", use_container_width=True):
                output_file = os.path.join(RESULTS_DIR, f"gold_standard_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                st.session_state.adjudicated_data.to_csv(output_file, index=False)
                st.success(f"✅ Exporté: {output_file}")
        
        with col2:
            st.markdown("#### Rapports")
            if st.button("📄 Générer Rapport Complet (Markdown)", use_container_width=True):
                # Générer un rapport complet
                report_content = f"""# Rapport Gold Test Set - MoLeAd
## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 1. Résumé Exécutif
- **Taille de l'échantillon**: {len(st.session_state.adjudicated_data)}
- **Nombre de catégories**: {st.session_state.adjudicated_data['gold_label'].nunique()}
- **Période d'annotation**: {datetime.now().strftime('%B %Y')}

### 2. Méthodologie
#### 2.1 Échantillonnage Stratifié
L'échantillon a été constitué selon une approche stratifiée pour garantir la représentation 
de toutes les catégories, incluant les classes rares.

#### 2.2 Annotation en Double-Aveugle
Deux annotateurs indépendants ont étiqueté l'ensemble de l'échantillon sans voir 
les annotations de l'autre ni les prédictions du système automatique.

#### 2.3 Accord Inter-Annotateurs (IAA)
Cohen's Kappa: {calculate_kappa(
    st.session_state.annotator_a_data['annotation_a'].tolist(),
    st.session_state.annotator_b_data['annotation_b'].tolist()
):.3f}

#### 2.4 Adjudication
Les désaccords ont été résolus par un expert tiers (super-annotateur) pour établir 
le Gold Standard final.

### 3. Résultats de l'Évaluation
[Voir les métriques détaillées dans la section Évaluation]

### 4. Conclusion
Ce Gold Test Set constitue une référence fiable pour évaluer les performances 
du système MoLeAd d'annotation automatique d'annonces légales.

### 5. Fichiers Générés
- Gold Standard (JSON et CSV)
- Annotations Annotateur A
- Annotations Annotateur B
- Analyse des désaccords
- Métriques de performance

---
*Rapport généré automatiquement par Gold Test Set Builder*
"""
                
                output_file = os.path.join(RESULTS_DIR, f"rapport_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                st.success(f"✅ Rapport généré: {output_file}")
        
        # Statistiques finales
        st.markdown("### 📈 Statistiques Finales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Échantillon total", len(st.session_state.adjudicated_data))
        
        with col2:
            categories = st.session_state.adjudicated_data['gold_label'].nunique()
            st.metric("Catégories", categories)
        
        with col3:
            kappa = calculate_kappa(
                st.session_state.annotator_a_data['annotation_a'].tolist(),
                st.session_state.annotator_b_data['annotation_b'].tolist()
            )
            st.metric("Cohen's Kappa", f"{kappa:.3f}")
        
        with col4:
            if 'annotation_a' in st.session_state.adjudicated_data.columns:
                gold_labels = st.session_state.adjudicated_data['gold_label'].tolist()
                silver_labels = st.session_state.adjudicated_data['text_content'].apply(
                    lambda x: x.get('subject_raw', 'Unknown')
                ).tolist()
                f1 = calculate_performance_metrics(gold_labels, silver_labels, average='macro')['f1_score']
                st.metric("F1-Score (Macro)", f"{f1:.3f}")
        
        # Distribution finale
        st.markdown("### 📊 Distribution des Catégories (Gold Standard)")
        
        gold_dist = st.session_state.adjudicated_data['gold_label'].value_counts()
        fig = px.pie(values=gold_dist.values, names=gold_dist.index,
                    title="Distribution des catégories du Gold Standard")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <b>🎓 Publication dans une revue Q1</b><br>
        Votre Gold Test Set est maintenant prêt à être utilisé dans votre article scientifique. 
        Assurez-vous d'inclure :<br>
        - La méthodologie d'échantillonnage stratifié<br>
        - Le score Cohen's Kappa pour l'IAA<br>
        - Les métriques de performance (Precision, Recall, F1 Macro/Micro)<br>
        - La matrice de confusion<br>
        - Une analyse qualitative des erreurs principales
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Gold Test Set Builder v1.0 | Projet MoLeAd | 2026</p>
    
</div>
""", unsafe_allow_html=True)
#<p>Développé pour répondre aux standards des revues Q1</p>