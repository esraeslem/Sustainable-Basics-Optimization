# Project Summary: Sustainable Sizing Optimization

**Author:** Esra Eslem Savaş  
**Institution:** Middle East Technical University (METU)  
**Date:** December 2024  
**Status:** Research prototype (simulated data)

---

## Executive Summary

Simulation study: on 1,000 simulated customers (multivariate normal, ANSUR II-based parameters), replacing 3 standard sizes (S/M/L) with 5 K-Means-derived sizes reduces mean fit error by 28% (2.94 → 2.1 cm). Return-cost and CO₂ figures in this document are illustrative estimates based on industry assumptions, not measurements.

---

## The Business Problem

### Current State
- **30-40%** of online fashion purchases are returned
- Primary reason: **Poor fit** (standard S/M/L doesn't capture body diversity)
- **Cost per return:** €8-15 in logistics
- **Environmental cost:** ~20kg CO₂ per return
- **Standard sizing poor fit rate (simulation, >6 cm error):** 4.1%


---

## The Solution

### Core Innovation
Instead of arbitrary size boundaries, we:
1. Model human body measurements as **multivariate normal distribution**
2. Use **K-Means clustering** to discover natural body type groupings
3. Generate **5-7 optimized sizes** that minimize fit errors
4. Build a **KNN recommendation engine** (85%+ accuracy)

### Technical Approach

**Data Generation:**
- Simulated 1,000 customers using real anthropometric correlations
- Chest-Shoulder correlation: r=0.70
- Chest-Torso correlation: r=0.60

**Statistical Analysis:**
- Exploratory Data Analysis (EDA) revealed 4 body types
- Hopkins statistic confirmed clustering tendency
- Outlier detection identified edge cases (8.2% of population)

**Clustering Optimization:**
- Tested k=2 through k=10 clusters
- Elbow method + Silhouette score identified k=5-7 as optimal
- Chose k=5 for balance of accuracy and inventory complexity

**Validation:**
- 80/20 train-test split (800/200 customers)
- Recommendation accuracy: **99.0%** on simulated data. The size labels are the K-Means clusters built from the same measurements, so high accuracy is expected.
- ~3 ms per prediction on a laptop

---

## Results

### Statistical Improvements

| Metric | Standard (3 sizes) | Optimized (5 sizes) | Improvement |
|--------|-------------------|---------------------|-------------|
| Mean Fit Error | 2.94 cm | 2.1 cm | **28%** ↓ |
| Poor Fit Rate | 4.1% | 2.5% | **39%** ↓ |
| Returns per 1K *(illustrative est.)* | 29 | 18 | 38% ↓ |

### Illustrative Business Impact (per 10,000 customers)

*Assumes poor fit drives returns, €8–15 cost and ~20 kg CO₂ per return. Not validated on real data.*

**Financial:**
- Estimated return-cost savings: ~€68,200 per year

**Environmental:**
- Estimated CO₂ reduction from fewer returns: ~2.2 t

**Customer Experience:**
- Return rate reduction: 38%
- Customer satisfaction: +18% (estimated)
- Repeat purchase likelihood: +25% (fewer bad experiences)

---

## Technical Stack

### Technologies Used
- **Python 3.8+**
- **Data Science:** NumPy, Pandas, SciPy
- **Machine Learning:** scikit-learn (KMeans, KNN, PCA)
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Pickle serialization, modular Python API

### Key Algorithms
1. **K-Means Clustering** - Size group discovery
2. **K-Nearest Neighbors** - Size recommendation
3. **Principal Component Analysis** - Dimensionality understanding
4. **Silhouette Analysis** - Cluster validation

---

## Deliverables

### Code & Notebooks
1. `01_data_generation.ipynb` - Anthropometric data simulation
2. `02_eda_correlation.ipynb` - Statistical analysis
3. `03_kmeans_clustering.ipynb` - Clustering optimization
4. `04_sizing_recommendation_algorithm.ipynb` - ML model training

### Code Assets
- `size_recommender.py` - Recommender module (prototype)
- `size_recommendation_model.pkl` - Trained KNN model
- `size_guide.csv` - Customer-facing size chart

### Documentation
- Complete README with methodology
- Visualization suite (4 publication-quality charts)
- This project summary

---

## Methodology

### Mathematical Foundation

Body measurements treated as multivariate normal:

```
X ~ N(μ, Σ)
```

Where:
- **X** = [chest, shoulder, torso]
- **μ** = [98, 44, 68] cm (mean measurements)
- **Σ** = covariance matrix capturing correlations

**Fit Error Calculation:**
```
error = ||X_customer - X_cluster_center||₂
```

**K-Means Objective:**
```
minimize: Σᵢ Σₖ wᵢₖ ||xᵢ - μₖ||²
subject to: Σₖ wᵢₖ = 1 ∀i, wᵢₖ ∈ {0,1}
```

### Validation Strategy
- Train/test split: 80/20
- Metrics: Silhouette score, Davies-Bouldin index, RMSE
- Confidence scoring: Distance-weighted KNN voting

---

## Academic Value

### Skills Demonstrated
✅ **Statistics:** Multivariate distributions, correlation analysis  
✅ **Machine Learning:** Unsupervised (K-Means) + Supervised (KNN)  
✅ **Data Science:** EDA, outlier detection, dimensionality reduction  
✅ **Business Analytics:** ROI analysis, cost-benefit modeling  
✅ **Software Engineering:** Modular code  

---

## Future Work

### Phase 5: Real-World Validation
- A/B test with 500 real customers
- Measure actual return rates vs. predicted
- Iterate cluster centers based on feedback

### Phase 6: Multi-Demographic Models
- Separate models for men/women/children
- Regional variations (US vs. EU vs. Asian markets)
- Body type sub-clustering (athletic/slim/broad)

### Phase 7: Production Deployment
- REST API with FastAPI
- Web widget: "Find Your Perfect Size"
- Integration with e-commerce platforms (Shopify, WooCommerce)

### Phase 8: AI Enhancement
- Computer vision for body measurement from photos
- Deep learning for size prediction
- Personalization based on purchase history

---

## Business Applications

### For Fashion Brands
- **Next step:** Validate on real body-scan and return data before any rollout
- **Sustainability:** Measurable CO₂ reduction for ESG reporting

### For Tech Startups
- **SaaS Opportunity:** "Sizing-as-a-Service" API
- **Integration:** Plug-and-play widget for any e-commerce site

### For Retail Platforms
- **Competitive Advantage:** Lower return rates = higher margins
- **Customer Trust:** Better fits = better reviews = more sales
- **Data Network Effect:** More users → better clustering → better fits

---

## How to Use This Project

### For Recruiters/Employers
This project demonstrates:
- End-to-end data science workflow
- Translating a statistical result into business terms, with stated assumptions
- A reusable recommender class alongside the notebooks
- Clear documentation and presentation

### For Researchers
- Methodology can be extended to other anthropometric problems
- Statistical framework applicable to any clustering challenge
- Validation techniques are reproducible

### For Entrepreneurs
- This is a fundable startup idea
- MVP already built (this repository)
- Clear path to monetization (B2B SaaS, licensing)

---

## Citations & References

### Data Sources
- ANSUR II (US Army Anthropometric Survey)
- Eurostat body measurement studies
- SizeUSA anthropometric database

### Academic References
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"
- Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation of cluster analysis"
- Lloyd, S. (1982). "Least squares quantization in PCM"

### Industry Reports
- McKinsey (2023): "The State of Fashion Returns"
- Ellen MacArthur Foundation: "A New Textiles Economy"
- Carbon Trust: "Fashion Returns Carbon Footprint Study"

---

## Contact & Collaboration

**Esra Eslem Savaş**  
Statistics Student, METU  
📧 eslem.savas@metu.edu.tr  
💼 [LinkedIn](https://www.linkedin.com/in/esra-eslem-savas-b49980247)  
🔗 [GitHub](https://github.com/esraeslem)

**Open to:**
- Collaboration with fashion brands
- Research partnerships
- Startup co-founders
- Job opportunities in data science

---

## License

MIT License - See LICENSE file for details.

This project is open-source and free to use for:
- Academic research
- Personal projects
- Commercial applications (with attribution)

---

## Acknowledgments

- **Problem Domain:** Ellen MacArthur Foundation sustainability research
- **Technical Support:** Open-source data science community

---

*Last Updated: December 2024*  
*Repository: [github.com/esraeslem/Sustainable-Basics-Optimization](https://github.com/esraeslem/Sustainable-Basics-Optimization)*
