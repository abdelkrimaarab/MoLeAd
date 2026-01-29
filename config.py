# Configuration - Gold Test Set Builder

## Paramètres Personnalisables

### Échantillonnage
SAMPLE_SIZE = 1000                    # Nombre total d'échantillons
MIN_RARE_SAMPLES = 50                # Quota minimum pour classes rares
RARE_CLASS_THRESHOLD = 0.05          # Classes < 5% du total = rares
RANDOM_SEED = 42                     # Pour reproductibilité

### Catégories d'Annotation
CATEGORIES = ["Creation", "Modification"]

### Canonical Subjects for CREATION
CREATION_CANONICAL = {
    "تأسيس شركة ذات المسؤولية المحدودة ذات الشريك الوحيد",
    "تأسيس شركة ذات المسؤولية المحدودة",
    "تأسيس شركة المساهمة",
    "تأسيس شركة الأسهم المبسطة ذات الشريك الوحيد",
    "تأسيس شركة الأسهم المبسطة",
    "تأسيس شركة تضامن",
    "تأسيس شركة التوصية البسيطة",
    "تأسيس شركة التوصية بالأسهم",
    "تأسيس مجموعة ذات نفع اقتصادي",
    "تأسيس الصندوق الجماعي للتوظيف",
    "تأسيس شركة مدنية عقارية",
    "تأسيس غير محدد"
}

### Canonical Subjects for MODIFICATION
MODIFICATION_CANONICAL = {
    "تغيير تسمية الشركة",
    "إضافة مختصر تسمية",
    "إضافة تسمية تجارية أو شعار",
    "رفع رأسمال الشركة",
    "خفض رأسمال الشركة",
    "تعيين مسير جديد شخص ذاتي أو اعتباري",
    "تجديد مدة مزاولة مهام المسيرين",
    "تفويت الحصص الاجتماعية",
    "تعيين مراقبي الحسابات",
    "تمديد مدة الشركة",
    "استمرار نشاط الشركة",
    "تحويل الشكل القانوني للشركة",
    "تحويل المقر الاجتماعي للشركة",
    "إنشاء فرع أو وكالة داخل دائرة المحكمة",
    "إنشاء فرع أو وكالة خارج دائرة المحكمة",
    "تغيير عنوان فرع أو وكالة داخل دائرة المحكمة",
    "تغيير عنوان فرع أو وكالة خارج دائرة المحكمة",
    "إغلاق فرع أو وكالة داخل دائرة المحكمة",
    "إغلاق فرع أو وكالة خارج دائرة المحكمة",
    "تغيير نشاط الشركة",
    "وفاة شريك أو مساهم",
    "حل شركة",
    "قفل التصفية لشركة مساهمة تدعو الجمهور للاكتتاب",
    "انفصال أو إدماج",
    "عقد التسيير الحر",
    "انتهاء عقد تسيير حر لأصل تجاري",
    "إعلان متعدد القرارات",
    "استدراك خطأ",
    "تعيين رئيس مجلس الإدارة",
    "تعيين أعضاء مجلس الإدارة",
    "تعيين مدير عام",
    "تعيين ممثل قانوني للشركة",
    "ملاءمة النظام الأساسي للشركة",
    "تغيير السنة المالية",
    "استدعاء للجموع العامة",
    "تقليص هدف الشركة",
    "فسخ عقد تسيير حر لأصل تجاري",
    "تعيين متصرفين",
    "توسيع نشاط الشركة"
}

### Get all canonical subjects
def get_all_canonical_subjects():
    """Returns a list of all canonical subjects from both categories"""
    return list(CREATION_CANONICAL) + list(MODIFICATION_CANONICAL)

### Mapping Arabe-Français (optionnel)
CATEGORY_MAPPING = {
    "Creation": "Création",
    "Modification": "Modification",
}

### Chemins de Fichiers
DATA_PATH = "./data/sample_announcements.json"
RESULTS_DIR = "./results"

### Seuils de Qualité
MIN_KAPPA = 0.60                     # Kappa minimum acceptable
TARGET_KAPPA = 0.80                  # Kappa cible (excellent)
MIN_CONFIDENCE = 4.0                 # Confiance moyenne souhaitée
MAX_OTHER_PERCENTAGE = 0.05          # Max 5% de catégorie "Autre"

### Performance Metrics
TARGET_F1_MACRO = 0.70               # F1 Macro cible
TARGET_F1_MICRO = 0.75               # F1 Micro cible
EXCELLENT_F1 = 0.80                  # F1 considéré excellent

### Annotation
AVG_TIME_PER_ANNOTATION = 2          # Minutes (pour planning)
MAX_ANNOTATIONS_PER_SESSION = 100    # Avant pause recommandée
BREAK_DURATION = 10                  # Minutes de pause

### Export
EXPORT_FORMATS = ["json", "csv"]     # Formats d'export disponibles
INCLUDE_TIMESTAMPS = True            # Inclure dates dans exports
INCLUDE_CONFIDENCE = True            # Inclure niveaux de confiance

### Visualisation
COLOR_SCHEME = "Blues"               # Schéma de couleurs Plotly
CHART_HEIGHT = 600                   # Hauteur par défaut des graphiques
SHOW_PERCENTAGES = True              # Afficher % dans les graphiques
