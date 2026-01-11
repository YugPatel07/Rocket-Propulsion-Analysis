ROCKET PROPELLANT DATABASE - GETTING STARTED
============================================

This package contains a comprehensive global rocket propellant database with 70+ propellant 
combinations and complete analysis tools for PCA/PLS modeling.

📦 PACKAGE CONTENTS
===================

1. rocket_propellants_global_database.csv
   - 70 propellant combinations from around the world
   - 37 quantitative and qualitative features
   - Covers liquid bipropellants, monopropellants, solid propellants, and hybrids
   - Data from USA, Russia, China, India, Japan, Europe
   - Mix of operational, historical, and experimental propellants

2. data_dictionary.txt
   - Comprehensive variable definitions
   - Explanation of all 37 features
   - Data sources and methodology
   - Recommended analysis approaches
   - Visualization suggestions

3. propellant_analysis.py
   - Complete Python analysis script
   - Functions for PCA, PLS regression, clustering
   - Automated visualization generation
   - Propellant selection helper tools

4. This README file


🚀 QUICK START
===============

OPTION A: Load data in Python/Pandas
-------------------------------------
import pandas as pd
df = pd.read_csv('rocket_propellants_global_database.csv')
print(df.head())
print(df.columns)

OPTION B: Run complete analysis script
---------------------------------------
python propellant_analysis.py

This will:
✓ Load and explore the data
✓ Generate correlation matrix
✓ Perform PCA analysis (5 components)
✓ Create PLS regression model (predict Isp)
✓ Run K-means clustering
✓ Generate 4 visualization files
✓ Show example propellant selection

Output files created:
- pca_analysis_plots.png
- pls_analysis_plots.png
- clustering_plots.png
- correlation_matrix.png


📊 KEY FEATURES IN DATASET
===========================

Performance Metrics:
- Vacuum_Isp_sec (180-542s range)
- Sea Level_Isp_sec
- Density_Impulse (volumetric performance)
- Combustion_Temp_K
- Exhaust molecular weight

Physical Properties:
- Bulk_Density_g_cm3 (0.36-1.95)
- Oxidizer/Fuel densities
- Boiling/Freezing points
- Storage temperature

Economic Factors:
- Cost_Score (1-10 scale)
- Storability duration (1-1825 days)
- Handling complexity
- Infrastructure requirements

Safety Metrics:
- Toxicity_Score (0-10)
- Corrosivity_Score (0-10)
- Environmental_Impact_Score
- Hypergolic flag (yes/no)

Technical Capabilities:
- TRL (Technology Readiness Level 1-9)
- Flight_Heritage_Count
- Restartable/Throttleable flags
- Primary application


🔬 ANALYSIS METHODS INCLUDED
==============================

1. Principal Component Analysis (PCA)
   - Reduce dimensionality of 20+ features
   - Identify key variance drivers
   - Visualize propellant clusters in PC space
   - Understand feature correlations

2. Partial Least Squares (PLS) Regression
   - Predict performance (Isp) from properties
   - Calculate Variable Importance in Projection (VIP)
   - Identify critical predictors
   - Handle collinearity better than standard regression

3. K-Means Clustering
   - Group similar propellants
   - Performed on PCA-transformed data
   - Identify propellant families
   - Validate against known chemical families

4. Correlation Analysis
   - Pearson correlations between all numeric features
   - Identify multicollinearity issues
   - Find unexpected relationships


💡 EXAMPLE USE CASES
=====================

Use Case 1: Find high-performance storable propellants
-------------------------------------------------------
from propellant_analysis import find_optimal_propellants, load_data

df = load_data('rocket_propellants_global_database.csv')
results = find_optimal_propellants(df, {
    'min_isp': 300,           # High performance
    'max_toxicity': 6,        # Moderately safe
    'storable': True,         # Room temp storage
    'flight_proven': True,    # TRL >= 8
    'restartable': True       # Engine restart capability
})

Use Case 2: Compare cryogenic vs hypergolic propellants
--------------------------------------------------------
import pandas as pd
df = pd.read_csv('rocket_propellants_global_database.csv')

cryo = df[df['Chemical_Family'] == 'Cryogenic']
hyper = df[df['Chemical_Family'] == 'Hypergolic']

print("Cryogenic Avg Isp:", cryo['Vacuum_Isp_sec'].mean())
print("Hypergolic Avg Isp:", hyper['Vacuum_Isp_sec'].mean())
print("Cryogenic Avg Density:", cryo['Bulk_Density_g_cm3'].mean())
print("Hypergolic Avg Density:", hyper['Bulk_Density_g_cm3'].mean())

Use Case 3: Identify emerging green propellants
------------------------------------------------
green = df[df['Chemical_Family'] == 'Green_Monoprop']
print(green[['Propellant_Name', 'Vacuum_Isp_sec', 'Toxicity_Score_0_10', 
             'TRL_1_9', 'Notes']])

Use Case 4: Multi-objective optimization visualization
-------------------------------------------------------
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Vacuum_Isp_sec'], 
           df['Bulk_Density_g_cm3'], 
           df['Toxicity_Score_0_10'],
           c=df['Cost_Score_1_10'], 
           cmap='viridis', 
           s=100, 
           alpha=0.6)

ax.set_xlabel('Isp (sec)')
ax.set_ylabel('Density (g/cm³)')
ax.set_zlabel('Toxicity Score')
plt.colorbar(label='Cost Score')
plt.title('Multi-Objective Propellant Comparison')
plt.show()


📈 RECOMMENDED ANALYSIS WORKFLOW
=================================

Step 1: Data Exploration
-------------------------
- Load data
- Check distributions of key variables
- Identify missing values
- Explore categorical breakdowns by type/family/country

Step 2: Correlation Analysis
-----------------------------
- Create correlation matrix
- Identify highly correlated features (VIF > 10)
- Understand relationships between performance, cost, safety

Step 3: Dimensionality Reduction (PCA)
---------------------------------------
- Standardize all features
- Run PCA with 5+ components
- Examine scree plot (elbow method)
- Analyze loadings to interpret PCs
- Visualize propellants in PC space

Step 4: Predictive Modeling (PLS)
----------------------------------
- Choose target variable (e.g., Vacuum_Isp_sec)
- Build PLS model with 2-5 components
- Calculate VIP scores
- Identify most important predictors
- Validate with cross-validation

Step 5: Clustering & Classification
------------------------------------
- Run K-means on PC scores (try k=3-7)
- Analyze cluster characteristics
- Compare to known propellant families
- Identify outliers or unique propellants

Step 6: Multi-Criteria Decision Analysis
-----------------------------------------
- Define mission requirements
- Weight objectives (performance, cost, safety, etc.)
- Score propellants on each criterion
- Use Pareto frontiers or TOPSIS method
- Select top candidates


⚠️ IMPORTANT NOTES
===================

Missing Data:
- Some variables have NA values by design
  (e.g., no separate fuel/oxidizer for monopropellants)
- Consider analyzing by Propellant_Type separately
- Use median imputation carefully

Data Quality:
- Isp values are theoretical (from CEA calculations)
- Actual performance varies with engine design
- Cost/complexity scores are relative estimates
- Flight heritage counts are approximate

Outliers:
- Exotic propellants (fluorine-based, tripropellants) are extreme but real
- Include/exclude based on analysis goals
- Consider separate analysis of "practical" vs "theoretical" options

Scaling:
- Always standardize features before PCA/PLS
- Different units (seconds, Kelvin, g/cm³) need normalization


🔧 CUSTOMIZATION OPTIONS
=========================

Modify Analysis Script:
1. Change number of PCA components (line 337)
2. Select different features for analysis (lines 114-135)
3. Adjust number of clusters (line 541)
4. Change PLS target variable (line 356)
5. Include/exclude solid propellants (line 351)

Add Your Own Data:
1. Open CSV in Excel/Google Sheets
2. Add new rows following same format
3. Fill in all 37 columns
4. Save and re-run analysis

Create Custom Visualizations:
- Use matplotlib, seaborn, or plotly
- Leverage the df_filtered and X_pca outputs
- Combine with mission-specific constraints


📚 ADDITIONAL RESOURCES
========================

For deeper understanding:
- NASA CEA documentation: https://www.grc.nasa.gov/www/CEAWeb/
- RocketCEA tool: https://rocketcea.readthedocs.io
- JANNAF resources: https://www.jannaf.org
- Sutton & Biblarz: "Rocket Propulsion Elements" (textbook)

For statistical methods:
- Scikit-learn PCA guide: https://scikit-learn.org/stable/modules/decomposition.html
- PLS regression tutorial: https://scikit-learn.org/stable/modules/cross_decomposition.html


🤝 CONTRIBUTING & FEEDBACK
===========================

Found an error in the data?
- Cross-reference with authoritative sources (NASA, ESA, etc.)
- Submit corrections with citations

Want to add more propellants?
- Follow CSV format exactly
- Include all 37 variables
- Provide source references in Notes field

Suggestions for analysis methods?
- Open for incorporating new techniques
- Bayesian optimization, neural networks, etc.


📜 LICENSE & CITATION
=====================

Data Sources:
- NASA Technical Reports Server
- JANNAF Databases
- NIST Chemistry WebBook
- International space agency publications
- Peer-reviewed literature

This compilation: November 2025

For academic use, please cite:
- Primary sources for specific propellant data
- RocketCEA for computational values
- This database as aggregation tool


✅ QUICK CHECKS
===============

Installation check:
-------------------
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages installed!')"

Load data check:
----------------
python -c "import pandas as pd; df = pd.read_csv('rocket_propellants_global_database.csv'); print(f'Loaded {len(df)} propellants successfully!')"

Run analysis check:
-------------------
python propellant_analysis.py

Expected output: 4 PNG files + console output with statistics


🎯 SUCCESS CRITERIA
====================

Your PCA/PLS analysis is working well if:
✓ PC1-PC3 explain >70% of variance
✓ PCA loadings make physical sense (performance vs safety vs cost)
✓ Propellant clusters align with chemical families
✓ PLS R² > 0.80 for Isp prediction
✓ VIP scores highlight known important factors (molecular weight, combustion temp)


📞 SUPPORT
==========

For technical issues with:
- Python script: Check package versions, Python 3.8+
- Data interpretation: Consult data_dictionary.txt
- Domain questions: Reference aerospace engineering textbooks


Happy analyzing! 🚀

=========================================
Last Updated: November 2025
Version: 1.0
=========================================
