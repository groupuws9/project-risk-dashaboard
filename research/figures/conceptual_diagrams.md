# Conceptual Diagrams (Mermaid)

## Figure 2.1: Relationship between the research problem, requirements and artefact

```mermaid
graph TD
    RP[Research Problem<br/>Classify operational risk<br/>from WERS workplace data]
    
    RQ1[Research Question 1<br/>What features predict<br/>operational risk?]
    RQ2[Research Question 2<br/>Which model achieves<br/>best balanced accuracy?]
    RQ3[Research Question 3<br/>How do features influence<br/>risk predictions?]
    
    DR1[Design Requirement 1<br/>Leakage-free feature set]
    DR2[Design Requirement 2<br/>Imbalance-aware evaluation]
    DR3[Design Requirement 3<br/>Model explainability]
    DR4[Design Requirement 4<br/>Validated API contract]
    
    A1[Artefact 1<br/>Processed dataset<br/>14,448 workplaces]
    A2[Artefact 2<br/>Trained models<br/>5 classifiers]
    A3[Artefact 3<br/>SHAP explanations<br/>Global + local]
    A4[Artefact 4<br/>Flask dashboard<br/>Prediction API]
    
    RP --> RQ1
    RP --> RQ2
    RP --> RQ3
    
    RQ1 --> DR1
    RQ2 --> DR2
    RQ3 --> DR3
    RQ2 --> DR4
    
    DR1 --> A1
    DR2 --> A2
    DR3 --> A3
    DR4 --> A4
    
    style RP fill:#e1f5ff
    style RQ1 fill:#fff4e1
    style RQ2 fill:#fff4e1
    style RQ3 fill:#fff4e1
    style DR1 fill:#f0e1ff
    style DR2 fill:#f0e1ff
    style DR3 fill:#f0e1ff
    style DR4 fill:#f0e1ff
    style A1 fill:#e1ffe1
    style A2 fill:#e1ffe1
    style A3 fill:#e1ffe1
    style A4 fill:#e1ffe1
```

## Figure 3.1: Research and evaluation workflow

```mermaid
graph TD
    START[Start<br/>Raw WERS data]
    
    STEP1[01_build_dataset.py<br/>Merge wirsts10 + wirsts25<br/>Construct turnover target]
    STEP2[02_exploratory_analysis.py<br/>EDA + correlation analysis]
    STEP3[03_feature_exploration.py<br/>EFA factor analysis]
    STEP4[04_model_training.py<br/>Train 5 models<br/>5-fold CV]
    STEP5[05_imbalance_tuning.py<br/>SMOTE/SMOTE-ENN variants]
    STEP6[06_model_evaluation.py<br/>Test set evaluation<br/>Confusion matrices]
    STEP7[07_shap_explanation.py<br/>SHAP global + local]
    STEP8[08_factor_analysis.py<br/>EFA scoring pipeline]
    
    DASH[dashboard/app.py<br/>Flask prediction API]
    TESTS[tests/test_api.py<br/>System tests ST1-ST19]
    
    RESULTS[research/tables/<br/>research/figures/]
    
    START --> STEP1
    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> STEP4
    STEP4 --> STEP5
    STEP4 --> STEP6
    STEP4 --> STEP7
    STEP3 --> STEP8
    
    STEP4 --> DASH
    DASH --> TESTS
    
    STEP1 --> RESULTS
    STEP4 --> RESULTS
    STEP5 --> RESULTS
    STEP6 --> RESULTS
    STEP7 --> RESULTS
    TESTS --> RESULTS
    
    style START fill:#e1f5ff
    style STEP1 fill:#fff4e1
    style STEP2 fill:#fff4e1
    style STEP3 fill:#fff4e1
    style STEP4 fill:#ffe1e1
    style STEP5 fill:#ffe1e1
    style STEP6 fill:#ffe1e1
    style STEP7 fill:#ffe1e1
    style STEP8 fill:#fff4e1
    style DASH fill:#e1ffe1
    style TESTS fill:#e1ffe1
    style RESULTS fill:#f0e1ff
```

## Figure 3.2: Logical artefact architecture

```mermaid
graph TB
    subgraph DATA_LAYER[Data Layer]
        RAW[Raw WERS Data<br/>wirsts10.tab<br/>wirsts25.tab]
        PROC[Processed Dataset<br/>wers_operational_risk_merged.csv<br/>14,448 x 53]
        META[Metadata<br/>dataset_metadata.json<br/>model_metadata.json]
    end
    
    subgraph MODEL_LAYER[Model Layer]
        PIPE[Canonical Cleaning<br/>Missing-tokens → Stringify → Strip → "MISSING"<br/>+ Rare → "OTHER_RARE"<br/>16 _MISSING one-hot]
        EFA[EFA Scoring<br/>4 factor scores]
        MODELS[Trained Models<br/>logreg, rf, xgboost, lgbm, mlp]
        BEST[Best Model<br/>logreg.pkl]
    end
    
    subgraph API_LAYER[API Layer]
        VALID[Pydantic Validation<br/>OperationalRiskInput]
        PRED[Prediction Endpoint<br/>/api/predict]
        WHATIF[What-If Lab<br/>/api/whatif]
        INSIGHTS[Insights Endpoint<br/>/api/insights]
    end
    
    subgraph UI_LAYER[UI Layer]
        DASH[Flask Dashboard<br/>index.html]
        FORMS[Prediction Forms<br/>Risk Prediction<br/>What-If Simulator]
        CHARTS[Visualizations<br/>SHAP plots<br/>Feature importance]
    end
    
    RAW --> PROC
    PROC --> META
    PROC --> PIPE
    META --> PIPE
    PIPE --> EFA
    PIPE --> MODELS
    EFA --> MODELS
    MODELS --> BEST
    
    BEST --> PRED
    VALID --> PRED
    VALID --> WHATIF
    BEST --> WHATIF
    MODELS --> INSIGHTS
    META --> INSIGHTS
    
    PRED --> DASH
    WHATIF --> DASH
    INSIGHTS --> DASH
    DASH --> FORMS
    DASH --> CHARTS
    
    style DATA_LAYER fill:#e1f5ff
    style MODEL_LAYER fill:#fff4e1
    style API_LAYER fill:#ffe1e1
    style UI_LAYER fill:#e1ffe1
```

## Figure 4.1: Implemented training pipeline

```mermaid
graph LR
    subgraph INPUT[Input]
        CSV[wers_operational_risk_merged.csv]
    end
    
    subgraph PREPROCESS[Preprocessing]
        CLEAN[Sentinel Cleaning<br/>Missing codes → NaN]
        SPLIT[Stratified Split<br/>80/20 train/test]
        IMPUTE[Imputation<br/>Median (numeric)<br/>Mode (categorical)]
        SCALE[Scaling<br/>StandardScaler]
        ENCODE[Encoding<br/>OneHotEncoder]
        EFA[EFA Factors<br/>4 latent scores]
    end
    
    subgraph CV[Cross-Validation]
        SKFOLD[StratifiedKFold<br/>5-fold, shuffle=True]
        TRAIN[Train on 4 folds]
        VAL[Validate on 1 fold]
    end
    
    subgraph MODELS[Model Training]
        LR[Logistic Regression]
        RF[Random Forest]
        XGB[XGBoost]
        LGBM[LightGBM]
        MLP[MLP]
    end
    
    subgraph SELECTION[Model Selection]
        METRICS[Metrics<br/>Balanced Accuracy<br/>Macro-F1]
        SELECT[Select Best<br/>Max CV balanced acc]
    end
    
    subgraph OUTPUT[Output]
        PKL[best_model.pkl]
        META[model_metadata.json]
        TABLES[model_comparison.csv<br/>evaluation_summary.csv]
    end
    
    CSV --> CLEAN
    CLEAN --> SPLIT
    SPLIT --> IMPUTE
    IMPUTE --> SCALE
    IMPUTE --> ENCODE
    SCALE --> EFA
    ENCODE --> EFA
    EFA --> SKFOLD
    
    SKFOLD --> TRAIN
    TRAIN --> VAL
    VAL --> LR
    VAL --> RF
    VAL --> XGB
    VAL --> LGBM
    VAL --> MLP
    
    LR --> METRICS
    RF --> METRICS
    XGB --> METRICS
    LGBM --> METRICS
    MLP --> METRICS
    
    METRICS --> SELECT
    SELECT --> PKL
    SELECT --> META
    METRICS --> TABLES
    
    style INPUT fill:#e1f5ff
    style PREPROCESS fill:#fff4e1
    style CV fill:#f0e1ff
    style MODELS fill:#ffe1e1
    style SELECTION fill:#e1ffe1
    style OUTPUT fill:#f0e1ff
```

## Figure 4.2: Runtime inference architecture

```mermaid
sequenceDiagram
    participant C as Client
    participant V as Pydantic Validator
    participant P as Predictor
    participant M as Model Pipeline
    participant E as EFA Scorer
    
    C->>V: POST /api/predict<br/>JSON payload
    V->>V: Validate schema<br/>Type/range checks
    alt Invalid
        V-->>C: 400 Error<br/>MISSING_FIELD/INVALID_TYPE/OUT_OF_RANGE
    else Valid
        V->>P: Pass validated data
        P->>E: Compute EFA factors
        E-->>P: 4 factor scores
        P->>M: Preprocess + predict (canonical: replace tokens → stringify → strip → "MISSING" → rare→"OTHER_RARE" → 16 _MISSING one-hot)
        M->>M: Replace tokens → stringify → strip → "MISSING" → rare→"OTHER_RARE" → 16 _MISSING one-hot → predict
        M->>M: Model inference
        M-->>P: risk_level + probabilities
        P->>P: Compute SHAP (optional)
        P-->>C: 200 Response<br/>risk_level + prob + local_explanation
    end
```

## Figure 4.3: Flask API integration

```mermaid
graph TD
    subgraph CLIENT[Client]
        WEB[Web Browser<br/>Dashboard UI]
        API[API Client<br/>External requests]
    end
    
    subgraph FLASK[Flask Application]
        CORS[CORS Middleware]
        RATE[Rate Limiter<br/>120 req/60s]
        LOG[Request Logger<br/>X-Request-ID]
        
        subgraph ENDPOINTS[Endpoints]
            PRED[/api/predict<br/>/predict]
            WHATIF[/api/whatif<br/>/simulate]
            DATA[/api/data/summary<br/>/data]
            FEAT[/api/features]
            INS[/api/insights]
        end
        
        subgraph VALIDATION[Validation Layer]
            PYD[Pydantic Schema<br/>OperationalRiskInput]
            ERR[Error Handler<br/>Structured responses]
        end
        
        subgraph PREDICTION[Prediction Layer]
            PREDICTOR[OperationalRiskPredictor]
            MODEL[best_model.pkl]
            SHAP[SHAP Explainer]
        end
    end
    
    subgraph STORAGE[Storage]
        MDATA[model_metadata.json]
        SHAP_DATA[shap_importance.csv<br/>shap_local.json]
        CSV[wers_operational_risk_merged.csv]
    end
    
    WEB --> CORS
    API --> CORS
    
    CORS --> RATE
    RATE --> LOG
    LOG --> ENDPOINTS
    
    PRED --> PYD
    WHATIF --> PYD
    DATA --> ENDPOINTS
    FEAT --> ENDPOINTS
    INS --> ENDPOINTS
    
    PYD --> PREDICTOR
    PREDICTOR --> MODEL
    PREDICTOR --> SHAP
    
    MODEL --> MDATA
    SHAP --> SHAP_DATA
    DATA --> CSV
    INS --> MDATA
    FEAT --> SHAP_DATA
    
    PYD --> ERR
    ERR --> CORS
    
    style CLIENT fill:#e1f5ff
    style FLASK fill:#fff4e1
    style STORAGE fill:#e1ffe1
    style ENDPOINTS fill:#ffe1e1
    style VALIDATION fill:#f0e1ff
    style PREDICTION fill:#e1ffe1
```
