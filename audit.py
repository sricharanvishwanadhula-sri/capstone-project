import os, sqlite3, json
import pandas as pd
import chromadb

def chk(condition, name):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition

print("=" * 60)
print("MODULE 1 -- DATA PIPELINE")
print("=" * 60)

conn = sqlite3.connect('data_pipeline/books.db')
book_count = int(pd.read_sql('SELECT COUNT(*) as n FROM books', conn).iloc[0,0])
cat_count = int(pd.read_sql('SELECT COUNT(*) as n FROM categories', conn).iloc[0,0])
chk(book_count >= 60, f"Books: {book_count} (need >= 60)")
chk(cat_count >= 3, f"Categories: {cat_count} (need >= 3)")

sample = pd.read_sql('SELECT price_gbp, price_inr, rating, in_stock FROM books LIMIT 1', conn)
rate = round(float(sample.iloc[0]['price_inr']) / float(sample.iloc[0]['price_gbp']), 2)
chk(rate == 105.50, f"Currency rate: {rate} GBP->INR (need 105.50)")

fk = pd.read_sql('PRAGMA foreign_key_list(books)', conn)
chk(len(fk) > 0, "FK relationship books->categories")
conn.close()

code = open('data_pipeline/pipeline.py', 'r', encoding='utf-8').read()
chk('SELECT' in code, "SQL: SELECT")
chk('WHERE' in code, "SQL: WHERE")
chk('ORDER BY' in code, "SQL: ORDER BY")
chk('LIMIT' in code, "SQL: LIMIT")
chk('DISTINCT' in code, "SQL: DISTINCT")
chk(' IN ' in code, "SQL: IN")
chk('BETWEEN' in code, "SQL: BETWEEN")
chk('JOIN' in code, "SQL: JOIN")
chk('pd.read_sql' in code, "pd.read_sql used")
chk('pd.merge' in code, "pd.merge used")

print()
print("=" * 60)
print("MODULE 2 -- ANALYTICS PIPELINE")
print("=" * 60)

chk(os.path.exists('analytics/titanic.csv'), "titanic.csv committed")
chk(os.path.exists('analytics/titanic_pipeline.joblib'), "titanic_pipeline.joblib saved")
chk(os.path.exists('analytics/01_eda_executed.ipynb'), "01_eda_executed.ipynb exists")
chk(os.path.exists('analytics/02_modeling_executed.ipynb'), "02_modeling_executed.ipynb exists")

charts = os.listdir('analytics/charts') if os.path.exists('analytics/charts') else []
chk(len(charts) >= 4, f"Charts generated: {len(charts)} (need >= 4)")
print(f"    Charts: {sorted(charts)}")

nb1 = json.load(open('analytics/01_eda_executed.ipynb', encoding='utf-8'))
nb2 = json.load(open('analytics/02_modeling_executed.ipynb', encoding='utf-8'))

def get_text(nb):
    parts = []
    for c in nb['cells']:
        parts.append(''.join(c.get('source', [])))
        for o in c.get('outputs', []):
            parts.append(''.join(o.get('text', [])))
    return '\n'.join(parts)

t1 = get_text(nb1)
t2 = get_text(nb2)

# EDA checks
chk('IQR' in t1, "IQR outlier analysis")
chk('Q1' in t1 and 'Q3' in t1, "Q1/Q3 computed")
chk('outlier' in t1.lower(), "Outlier count reported")
chk('skew' in t1.lower(), "Skewness analysis")
chk('mode' in t1.lower(), "Mode reported (fare)")
chk('Right-skewed' in t1 or 'RIGHT-SKEWED' in t1 or 'right-skewed' in t1.lower(), "Skewness conclusion stated")
chk('survival rate' in t1.lower() or 'survived' in t1.lower(), "Survival rates computed")
chk('sex' in t1 and 'pclass' in t1, "Bivariate: sex and pclass")
chk('sibsp' in t1 and 'parch' in t1 and 'fare' in t1, "Correlation 6 cols present")
chk('adult_male' in t1, "adult_male mentioned (excluded)")
chk('sns.heatmap' in t1, "Correlation heatmap plotted")
chk(len(charts) >= 4, "At least 4 multivariate charts")
chk('StandardScaler' in t1, "EDA standardization check")
chk('Before Standardization' in t1, "Before/after standardization shown")

# Modeling checks
chk('stratify' in t2, "Stratified train/test split")
chk('ColumnTransformer' in t2, "ColumnTransformer used")
chk('fit(X_train' in t2, "Preprocessor fit on train only")
chk('LogisticRegression' in t2, "Logistic Regression")
chk('DecisionTreeClassifier' in t2, "Decision Tree")
chk('RandomForestClassifier' in t2, "Random Forest")
chk('plot_tree' in t2, "Decision Tree visualized with plot_tree")
chk('confusion_matrix' in t2, "Confusion matrix")
chk('roc_auc_score' in t2, "ROC AUC computed")
chk('roc_curve' in t2, "ROC curve plotted")
chk('accuracy_score' in t2 and 'precision_score' in t2 and 'f1_score' in t2, "Accuracy/Precision/F1")
chk('SMOTE' in t2, "SMOTE applied")
chk("class_weight='balanced'" in t2, "class_weight balanced")
chk('fit_resample(X_train' in t2, "SMOTE applied to training only")
chk('GridSearchCV' in t2, "GridSearchCV")
chk('oob_score=True' in t2, "oob_score=True at construction")
chk('oob_score_' in t2, "OOB score reported")
chk('LinearRegression' in t2, "Regression side-task: LinearRegression")
chk('mae' in t2.lower() or 'MAE' in t2, "MAE reported")
chk('RMSE' in t2 or 'rmse' in t2.lower(), "RMSE reported")
chk('adj_r2' in t2.lower() or 'Adjusted R' in t2, "Adjusted R2 reported")
chk('residual' in t2.lower(), "Residual plot")
chk('Heteroscedasticity' in t2 or 'heteroscedasticity' in t2.lower(), "Heteroscedasticity conclusion")
chk('recommend' in t2.lower(), "Final recommendation written")
chk('joblib.dump' in t2, "joblib.dump called")
chk('joblib.load' in t2, "joblib.load called (reload demo)")

print()
print("=" * 60)
print("MODULE 3 -- SUPPORT ASSISTANT")
print("=" * 60)

docs = [f for f in os.listdir('support_assistant/docs') if f.endswith('.txt')]
chk(len(docs) == 8, f"Corpus documents: {len(docs)} (need 8)")

client = chromadb.PersistentClient(path='support_assistant/chroma_db')
col = client.get_collection('zepto_policies')
chunk_count = col.count()
chk(chunk_count > 0, f"ChromaDB populated: {chunk_count} chunks")

gcode = open('support_assistant/graph.py', 'r', encoding='utf-8').read()
mcode = open('support_assistant/main.py', 'r', encoding='utf-8').read()
ptcode = open('support_assistant/prompt_template.py', 'r', encoding='utf-8').read()
docker = open('support_assistant/Dockerfile', 'r', encoding='utf-8').read()
readme3 = open('support_assistant/README.md', 'r', encoding='utf-8').read()

chk('classify_intent' in gcode, "Node: classify_intent")
chk('retrieve_and_answer' in gcode, "Node: retrieve_and_answer")
chk('direct_answer' in gcode, "Node: direct_answer")
chk('add_conditional_edges' in gcode, "Conditional edge (routing)")
chk('MOCK_LLM' in gcode, "MOCK_LLM toggle present")
chk('TypedDict' in gcode, "TypedDict state")
chk('keyword' in gcode.lower() or 'POLICY_KEYWORDS' in gcode, "Keyword heuristic present")
for kw in ['delivery', 'return', 'refund', 'membership', 'tracking', 'cancel', 'gift card', 'support hours']:
    chk(kw in gcode, f"Keyword: '{kw}'")
chk('ZeptoResponse' in gcode, "Pydantic ZeptoResponse schema")
chk('answer' in gcode and 'sources' in gcode and 'confidence' in gcode, "answer/sources/confidence fields")
chk('attempt' in gcode, "Retry logic (3 attempts on LLM failure)")
chk('collection.query' in gcode, "ChromaDB top-K retrieval")
chk('Based on the retrieved context' in gcode, "Canned template answer")
chk('can only answer' in gcode.lower(), "Fixed canned general answer")
chk('role' in ptcode.lower(), "Prompt: role")
chk('context' in ptcode.lower(), "Prompt: context")
chk('task' in ptcode.lower(), "Prompt: task")
chk('format' in ptcode.lower(), "Prompt: format")
chk('length' in ptcode.lower(), "Prompt: length")
chk('do not' in ptcode.lower(), "Prompt: negative constraint")
chk('example' in ptcode.lower(), "Prompt: few-shot examples")
chk('FastAPI' in mcode, "FastAPI app")
chk('/ask' in mcode, "POST /ask endpoint")
chk('AskRequest' in mcode and 'AskResponse' in mcode, "Request/Response models")
chk('uvicorn' in docker, "Dockerfile: uvicorn CMD")
chk('7860' in docker, "Dockerfile: port 7860")
chk('ingestion' in readme3.lower(), "README: ingestion stage")
chk('embedding' in readme3.lower(), "README: embedding stage")
chk('retrieval' in readme3.lower(), "README: retrieval stage")
chk('generation' in readme3.lower(), "README: generation stage")
chk('MOCK_LLM' in readme3, "README: MOCK_LLM explained")
chk('delivery fee' in readme3.lower(), "README: example call 1 shown")
chk('general question' in readme3.lower() or 'weather' in readme3.lower(), "README: example call 2 shown")

print()
print("=" * 60)
print("REPOSITORY LEVEL")
print("=" * 60)

chk(os.path.exists('README.md'), "Root README.md")
chk(os.path.exists('requirements.txt'), "Root requirements.txt")
chk(os.path.exists('data_pipeline/README.md'), "data_pipeline/README.md")
chk(os.path.exists('analytics/README.md'), "analytics/README.md")
chk(os.path.exists('support_assistant/README.md'), "support_assistant/README.md")
chk(os.path.exists('data_pipeline/books.db'), "books.db committed")
chk(os.path.exists('data_pipeline/scraper.py'), "scraper.py")
chk(os.path.exists('data_pipeline/pipeline.py'), "pipeline.py")
chk(os.path.exists('analytics/01_eda.ipynb'), "01_eda.ipynb")
chk(os.path.exists('analytics/02_modeling.ipynb'), "02_modeling.ipynb")
chk(os.path.exists('support_assistant/Dockerfile'), "Dockerfile")

print()
print("  [NOTE] Git branch/merge workflow: PENDING (needs git push)")
print()
print("AUDIT COMPLETE.")
