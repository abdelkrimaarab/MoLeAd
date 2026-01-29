# Gold Test Set Builder - MoLeAd

## 📋 Description

Application Streamlit complète pour constituer un **Gold Test Set** (Vérité Terrain) conforme aux exigences des revues Q1. Cette application vous guide à travers tout le processus de création d'un ensemble de test de référence pour valider votre méthode d'annotation automatique (weak supervision).

## 🎯 Objectifs

Prouver que votre méthode d'annotation automatique MoLeAd est fiable en la comparant à une annotation humaine de haute qualité.

## 🚀 Installation

### Prérequis
- Python 3.8+
- Les données brutes dans `../data/dataset/legal_announcements.json`

### Dépendances

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

## 💻 Lancement de l'application

```bash
cd "Gold Test Set"
streamlit run gold_test_set_app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut.

## 📊 Workflow en 6 étapes

### 1️⃣ Échantillonnage Stratifié
- Sélection de **500 à 1 000 annonces**
- Approche stratifiée pour garantir la représentation des classes rares
- Quota minimum configurable pour les catégories minoritaires
- Évite le biais vers les classes majoritaires

**Recommandation** : 1 000 documents est un standard solide pour une publication Q1

### 2️⃣ Annotation en Double Aveugle

#### 2a. Annotateur A
- Interface d'annotation intuitive
- Annotation indépendante sans voir les prédictions automatiques
- Suivi de progression en temps réel
- Niveau de confiance et notes optionnelles

#### 2b. Annotateur B
- Même interface que l'Annotateur A
- **Totalement indépendant** - ne voit pas les annotations de A
- Garantit l'objectivité de l'évaluation

**Important** : Les deux annotateurs doivent être des experts juridiques ou des linguistes formés

### 3️⃣ Accord Inter-Annotateurs (IAA)

Calcul automatique de :
- **Cohen's Kappa** : Mesure standard de l'accord inter-annotateurs
- **Taux d'accord simple** : Pourcentage d'accords directs
- **Matrice de confusion** : Visualisation des désaccords
- **Identification automatique des conflits** à résoudre

**Objectif** : Kappa > 0.8 (accord "fort" ou "presque parfait")

### 4️⃣ Adjudication

- Interface dédiée pour le **super-annotateur** (expert tiers)
- Examine uniquement les désaccords entre A et B
- Décide de l'étiquette finale pour chaque conflit
- Justification optionnelle des décisions
- Construction du **Gold Standard** final

### 5️⃣ Évaluation des Performances

Comparaison automatique entre :
- **Silver Labels** : Annotations automatiques de MoLeAd
- **Gold Labels** : Annotations humaines validées

#### Métriques calculées :

**Macro-Averaged** (traite toutes les classes de façon égale) :
- Precision
- Recall
- F1-Score

**Micro-Averaged** (pondéré par la fréquence des classes) :
- Precision
- Recall
- F1-Score

**Globales** :
- Accuracy
- Matrice de confusion complète
- Rapport de classification détaillé par classe

**Analyse des erreurs** :
- Identification des erreurs spécifiques
- Visualisation des patterns d'erreurs
- Top 10 des types d'erreurs les plus fréquents

### 6️⃣ Export & Rapport

**Exports disponibles** :
- Gold Standard (JSON et CSV)
- Annotations individuelles (A et B)
- Désaccords pour analyse
- Métriques de performance

**Rapport final** :
- Rapport Markdown complet
- Méthodologie détaillée
- Résultats statistiques
- Prêt pour inclusion dans une publication scientifique

## 📁 Structure des fichiers générés

```
Gold Test Set/
├── gold_test_set_app.py          # Application principale
├── README.md                      # Ce fichier
└── results/                       # Dossier auto-créé
    ├── sampled_data_*.json        # Échantillon stratifié
    ├── annotator_a_*.json         # Annotations de A
    ├── annotator_b_*.json         # Annotations de B
    ├── disagreements_*.csv        # Liste des désaccords
    ├── gold_standard_final_*.json # Gold Standard final
    ├── gold_standard_final_*.csv  # Gold Standard (CSV)
    └── rapport_final_*.md         # Rapport complet
```

## 🎓 Standards pour publication Q1

### Méthodologie à inclure dans l'article

1. **Échantillonnage** :
   ```
   "Un échantillon stratifié de 1 000 annonces légales a été constitué, 
   garantissant une représentation équilibrée de toutes les catégories, 
   avec un quota minimum de 50 instances pour les classes rares."
   ```

2. **Annotation** :
   ```
   "Deux annotateurs experts indépendants ont étiqueté l'ensemble de 
   l'échantillon en double-aveugle, sans accès aux prédictions du système 
   automatique."
   ```

3. **IAA** :
   ```
   "L'accord inter-annotateurs, mesuré par le Cohen's Kappa, était de κ = X.XX, 
   indiquant un accord [substantiel/fort/presque parfait]."
   ```

4. **Adjudication** :
   ```
   "Les X désaccords ont été résolus par un expert tiers indépendant, 
   constituant le Gold Standard final de référence."
   ```

5. **Résultats** :
   ```
   "Comparé au Gold Standard, notre système MoLeAd a atteint une précision 
   macro-moyenne de X.XX, un rappel de X.XX et un F1-Score de X.XX."
   ```

### Tableaux et figures recommandés

1. **Tableau 1** : Distribution des catégories dans l'échantillon
2. **Tableau 2** : Matrice de confusion inter-annotateurs
3. **Tableau 3** : Métriques de performance par catégorie
4. **Figure 1** : Matrice de confusion Gold vs. Silver
5. **Figure 2** : Distribution des erreurs par type

## 🔍 Interprétation des résultats

### Cohen's Kappa
- **κ < 0.20** : Accord faible → Réviser la taxonomie
- **0.20 ≤ κ < 0.40** : Accord moyen → Améliorer les guidelines
- **0.40 ≤ κ < 0.60** : Accord modéré → Clarifier les cas ambigus
- **0.60 ≤ κ < 0.80** : Accord substantiel → Acceptable pour Q1
- **κ ≥ 0.80** : Accord fort/presque parfait → Excellent pour Q1

### F1-Score
- **F1 > 0.80** : Excellent système
- **0.70 ≤ F1 ≤ 0.80** : Bon système, marges d'amélioration
- **0.60 ≤ F1 < 0.70** : Système acceptable, nécessite optimisation
- **F1 < 0.60** : Système nécessite révision importante

## 📚 Références scientifiques

Pour justifier votre méthodologie dans l'article :

1. **Cohen's Kappa** :
   - Cohen, J. (1960). "A coefficient of agreement for nominal scales"

2. **Échantillonnage stratifié** :
   - Cochran, W. G. (1977). "Sampling techniques"

3. **Annotation en double-aveugle** :
   - Artstein, R., & Poesio, M. (2008). "Inter-coder agreement for computational linguistics"

4. **Weak Supervision** :
   - Ratner, A., et al. (2017). "Snorkel: Rapid training data creation with weak supervision"

## 🛠️ Personnalisation

### Modifier les catégories d'annotation

Dans `gold_test_set_app.py`, ligne ~380 et ~450, modifiez :
```python
options=["", "Création", "Modification", "Dissolution", "Fusion/Scission", "Autre"]
```

### Ajuster les seuils d'échantillonnage

Dans la fonction `stratified_sampling()` :
```python
threshold = len(df) * 0.05  # 5% du total = classe rare
```

### Personnaliser les exports

Modifiez les fonctions d'export dans la section "Page 6: Export"

## ⚠️ Bonnes pratiques

1. **Formation des annotateurs** :
   - Organisez une session de formation avant l'annotation
   - Fournissez un guide d'annotation détaillé
   - Utilisez des exemples pour chaque catégorie

2. **Qualité avant quantité** :
   - Mieux vaut 500 annotations de haute qualité que 2000 médiocres
   - Encouragez les pauses régulières

3. **Documentation** :
   - Documentez toutes les décisions méthodologiques
   - Conservez les justifications d'adjudication
   - Notez les cas difficiles pour discussion

4. **Validation croisée** :
   - Envisagez plusieurs annotateurs pour les cas très ambigus
   - Discutez des désaccords systématiques en équipe

## 🐛 Dépannage

### Erreur de chargement des données
```
Vérifiez que le fichier legal_announcements.json existe dans :
../data/dataset/legal_announcements.json
```

### Application lente
```
L'application peut être lente avec de très gros fichiers.
Envisagez de créer un échantillon préliminaire plus petit.
```

## 📞 Support

Pour toute question ou problème :
1. Consultez la documentation Streamlit : https://docs.streamlit.io
2. Vérifiez les logs dans le terminal
3. Assurez-vous que toutes les dépendances sont installées

## 📜 Licence

Ce projet fait partie du système MoLeAd pour l'extraction et la classification automatique d'annonces légales.

## ✨ Améliorations futures

- [ ] Support multi-langues
- [ ] Export au format LaTeX pour articles
- [ ] Intégration avec des outils d'annotation comme Label Studio
- [ ] Calcul automatique de la taille d'échantillon optimale
- [ ] Support du Fleiss' Kappa pour >2 annotateurs
- [ ] Génération automatique de graphiques pour publication
- [ ] API REST pour intégration dans d'autres systèmes

---

**Développé pour le projet MoLeAd - 2026**

*Conforme aux standards des revues Q1 en NLP et Machine Learning*
