"""
Script d'analyse des résultats du Gold Test Set
Génère des statistiques et visualisations supplémentaires
"""

import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_gold_standard(filepath):
    """Charge le fichier Gold Standard"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analyze_distribution(df, label_column='gold_label'):
    """Analyse la distribution des catégories"""
    distribution = df[label_column].value_counts()
    
    print("=" * 60)
    print("DISTRIBUTION DES CATÉGORIES")
    print("=" * 60)
    for category, count in distribution.items():
        percentage = count / len(df) * 100
        print(f"{category:20s}: {count:5d} ({percentage:5.2f}%)")
    print("-" * 60)
    print(f"{'TOTAL':20s}: {len(df):5d} (100.00%)")
    print("=" * 60)
    
    return distribution

def calculate_agreement_metrics(df, col_a='annotation_a', col_b='annotation_b'):
    """Calcule les métriques d'accord inter-annotateurs"""
    labels_a = df[col_a].dropna().tolist()
    labels_b = df[col_b].dropna().tolist()
    
    # Assurer même longueur
    min_len = min(len(labels_a), len(labels_b))
    labels_a = labels_a[:min_len]
    labels_b = labels_b[:min_len]
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(labels_a, labels_b)
    
    # Accord simple
    agreements = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    simple_agreement = agreements / len(labels_a)
    
    # Désaccords
    disagreements = len(labels_a) - agreements
    
    print("\n" + "=" * 60)
    print("ACCORD INTER-ANNOTATEURS")
    print("=" * 60)
    print(f"Cohen's Kappa        : {kappa:.4f}")
    print(f"Accord simple        : {simple_agreement:.2%}")
    print(f"Accords              : {agreements}")
    print(f"Désaccords           : {disagreements}")
    print(f"Total annotations    : {len(labels_a)}")
    print("=" * 60)
    
    # Interprétation
    if kappa > 0.8:
        interpretation = "Presque parfait (> 0.8)"
    elif kappa > 0.6:
        interpretation = "Substantiel (0.6-0.8)"
    elif kappa > 0.4:
        interpretation = "Modéré (0.4-0.6)"
    elif kappa > 0.2:
        interpretation = "Moyen (0.2-0.4)"
    else:
        interpretation = "Faible (< 0.2)"
    
    print(f"\nInterprétation : {interpretation}")
    
    return {
        'kappa': kappa,
        'simple_agreement': simple_agreement,
        'agreements': agreements,
        'disagreements': disagreements
    }

def evaluate_performance(df, gold_col='gold_label', pred_col='subject_raw'):
    """Évalue les performances du système automatique"""
    
    # Extraire les predictions
    if pred_col == 'subject_raw':
        predictions = df['text_content'].apply(
            lambda x: x.get('subject_raw', 'Unknown') if isinstance(x, dict) else 'Unknown'
        ).tolist()
    else:
        predictions = df[pred_col].tolist()
    
    gold_labels = df[gold_col].tolist()
    
    # Filtrer les valeurs nulles
    valid_indices = [i for i in range(len(gold_labels)) 
                    if gold_labels[i] is not None and predictions[i] is not None]
    
    gold_filtered = [gold_labels[i] for i in valid_indices]
    pred_filtered = [predictions[i] for i in valid_indices]
    
    print("\n" + "=" * 60)
    print("ÉVALUATION DES PERFORMANCES")
    print("=" * 60)
    
    # Rapport de classification
    report = classification_report(gold_filtered, pred_filtered, 
                                   output_dict=True, zero_division=0)
    
    print("\nMÉTRIQUES GLOBALES")
    print("-" * 60)
    print(f"Accuracy            : {report['accuracy']:.4f}")
    print(f"Macro Avg Precision : {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall    : {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-Score  : {report['macro avg']['f1-score']:.4f}")
    print(f"Micro Avg Precision : {report['weighted avg']['precision']:.4f}")
    print(f"Micro Avg Recall    : {report['weighted avg']['recall']:.4f}")
    print(f"Micro Avg F1-Score  : {report['weighted avg']['f1-score']:.4f}")
    
    print("\nPAR CATÉGORIE")
    print("-" * 60)
    print(f"{'Catégorie':20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-" * 60)
    
    for category, metrics in report.items():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{category:20s} {metrics['precision']:10.4f} {metrics['recall']:10.4f} "
                  f"{metrics['f1-score']:10.4f} {int(metrics['support']):10d}")
    
    print("=" * 60)
    
    return report

def analyze_errors(df, gold_col='gold_label', pred_col='subject_raw'):
    """Analyse détaillée des erreurs"""
    
    # Extraire les predictions
    if pred_col == 'subject_raw':
        predictions = df['text_content'].apply(
            lambda x: x.get('subject_raw', 'Unknown') if isinstance(x, dict) else 'Unknown'
        ).tolist()
    else:
        predictions = df[pred_col].tolist()
    
    gold_labels = df[gold_col].tolist()
    
    errors = []
    for i, (gold, pred) in enumerate(zip(gold_labels, predictions)):
        if gold != pred:
            errors.append({
                'index': i,
                'id': df.iloc[i]['id'],
                'gold': gold,
                'prediction': pred,
                'company': df.iloc[i]['text_content'].get('company_name', 'N/A')
            })
    
    print("\n" + "=" * 60)
    print("ANALYSE DES ERREURS")
    print("=" * 60)
    print(f"Total erreurs       : {len(errors)}")
    print(f"Taux d'erreur       : {len(errors)/len(df):.2%}")
    
    # Types d'erreurs les plus fréquents
    error_types = Counter([f"{e['gold']} → {e['prediction']}" for e in errors])
    
    print("\nTOP 10 TYPES D'ERREURS")
    print("-" * 60)
    for error_type, count in error_types.most_common(10):
        percentage = count / len(errors) * 100 if len(errors) > 0 else 0
        print(f"{error_type:40s}: {count:4d} ({percentage:5.2f}%)")
    
    print("=" * 60)
    
    return errors

def generate_latex_tables(df, gold_col='gold_label', output_file='tables.tex'):
    """Génère des tableaux LaTeX pour publication"""
    
    latex_content = []
    
    # Table 1: Distribution
    latex_content.append("% Table 1: Distribution des Catégories")
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Distribution des catégories dans le Gold Test Set}")
    latex_content.append("\\begin{tabular}{lrr}")
    latex_content.append("\\toprule")
    latex_content.append("Catégorie & Nombre & Pourcentage \\\\")
    latex_content.append("\\midrule")
    
    distribution = df[gold_col].value_counts()
    for category, count in distribution.items():
        percentage = count / len(df) * 100
        latex_content.append(f"{category} & {count} & {percentage:.1f}\\% \\\\")
    
    latex_content.append("\\midrule")
    latex_content.append(f"Total & {len(df)} & 100.0\\% \\\\")
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("\n")
    
    # Sauvegarder
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"\n✅ Tableaux LaTeX générés dans {output_file}")

def plot_confusion_matrix(df, gold_col='gold_label', pred_col='subject_raw', 
                         output_file='confusion_matrix.png'):
    """Génère une matrice de confusion"""
    
    if pred_col == 'subject_raw':
        predictions = df['text_content'].apply(
            lambda x: x.get('subject_raw', 'Unknown') if isinstance(x, dict) else 'Unknown'
        ).tolist()
    else:
        predictions = df[pred_col].tolist()
    
    gold_labels = df[gold_col].tolist()
    
    # Créer la matrice
    unique_labels = sorted(list(set(gold_labels + predictions)))
    cm = confusion_matrix(gold_labels, predictions, labels=unique_labels)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Matrice de Confusion: Gold Standard vs. Prédictions Automatiques', 
              fontsize=14, pad=20)
    plt.xlabel('Prédiction (Silver Labels)', fontsize=12)
    plt.ylabel('Vérité Terrain (Gold Labels)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matrice de confusion sauvegardée dans {output_file}")

def main():
    """Fonction principale"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <chemin_vers_gold_standard.json>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print("\n" + "=" * 60)
    print("ANALYSE DU GOLD TEST SET - MOLEAD")
    print("=" * 60)
    print(f"Fichier: {filepath}\n")
    
    # Charger les données
    df = load_gold_standard(filepath)
    print(f"✅ {len(df)} annotations chargées\n")
    
    # 1. Analyse de distribution
    analyze_distribution(df)
    
    # 2. Métriques d'accord (si disponible)
    if 'annotation_a' in df.columns and 'annotation_b' in df.columns:
        calculate_agreement_metrics(df)
    
    # 3. Évaluation des performances
    if 'gold_label' in df.columns:
        evaluate_performance(df)
        
        # 4. Analyse des erreurs
        errors = analyze_errors(df)
        
        # 5. Générer tableaux LaTeX
        generate_latex_tables(df, output_file='tables_latex.tex')
        
        # 6. Matrice de confusion
        plot_confusion_matrix(df, output_file='confusion_matrix.png')
    
    print("\n" + "=" * 60)
    print("ANALYSE TERMINÉE")
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - tables_latex.tex")
    print("  - confusion_matrix.png")
    print("\n")

if __name__ == "__main__":
    main()
