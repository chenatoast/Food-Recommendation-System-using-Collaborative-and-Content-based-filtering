# Smart Food Recommendation System (Meal HARMONY)

![Meal HARMONY Logo](https://via.placeholder.com/150)

## Project Overview

Meal HARMONY is an intelligent food recommendation system that solves the everyday dilemma of "Khane mai kya banau?" (What should I cook?). Our solution uses machine learning to provide personalized meal suggestions based on user preferences, available ingredients, meal context, and other factors.

🔗 **Project Workflow:** [View our interactive design board](https://www.figma.com/board/1gTiz4wiUAzh9tDhEZxGfH/Meal-HARMONY?node-id=0-1&t=oz8yOiTEXQjjANcF-1)

## Problem Statement

Modern households face the daily challenge of meal planning, often resulting in:

- Repetitive, uninspiring food choices
- Unhealthy convenience options
- Excessive time spent on meal planning
- Difficulty balancing nutritional needs, preferences, and constraints

Our system addresses these challenges by creating a personalized, contextually aware recommendation engine that learns from user feedback and adapts to household constraints.

## Core Features

- **Personalized Recommendations:** Tailored suggestions based on user preference history
- **Ingredient-Based Filtering:** Recommends dishes that can be prepared with available ingredients
- **Meal-Time Context:** Adapts recommendations based on specific meal times (breakfast, lunch, dinner, snacks)
- **Continuous Learning:** Improves suggestions through user feedback and rating analysis
- **Variety Management:** Prevents recommendation repetition by tracking recently selected dishes

## Filtering Techniques

Our system employs a multi-stage filtering pipeline that combines various techniques:

### 1. User Preference Filtering

- Identifies dishes with low ratings (< 3) as improvement candidates
- Prioritizes recommendations based on highest potential for user satisfaction
- Excludes recently selected dishes to ensure variety

### 2. Ingredient Availability Filtering

- Uses set theory to verify required ingredients are a subset of available ingredients
- Implements a fallback mechanism using similarity-based recommendations when exact matches aren't available
- Mathematical representation:
  ```
  dish_ingredients ⊆ available_ingredients
  ```

### 3. Contextual Filtering (Meal-Time)

- Filters dishes based on appropriateness for selected meal time (breakfast, lunch, dinner, snacks)
- Uses a binary matrix to determine meal-time compatibility

### 4. Cosine Similarity Fallback

- When strict filtering yields insufficient results, the system relaxes constraints using similarity measures
- Finds dishes similar to candidate recommendations but with available ingredients
- Provides "next best" recommendations when exact matches aren't possible

## Mathematical & ML Principles

### 1. Cosine Similarity

**Purpose:** Measure similarity between dishes based on user rating patterns

**Formula:**

```
similarity(A,B) = cos(θ) = (A·B)/(||A||·||B||)
```

**Implementation:** Creates a dish similarity matrix with shape (n_dishes, n_dishes)

**Advantage:** Scale-invariant measure that focuses on rating patterns rather than absolute values

### 2. Linear Regression for Cold Start Problem

**Purpose:** Generate initial ratings for new users

**Model:** y = Xβ + ε

**Implementation:**

```python
linear_model = LinearRegression().fit(y, x)
```

where y is user indices and x is the rating matrix

**Process:** Captures rating trends across existing users to make predictions for new users

### 3. Neighborhood-Based Collaborative Filtering

**Purpose:** Identify the most similar dishes to improve recommendations

**Concept:** K-nearest neighbors in similarity space

**Implementation:**

```python
sorted_indices = np.argsort(item_similarity_scores)[::-1]
neighborhood = sorted_indices[1:neighborhood_size+1]
```

### 4. Dynamic Rating Adjustment

**Purpose:** Propagate rating changes to similar dishes

**Formula:**

```
new_rating = current_rating + (0.1 * (recommendation_rating - 3)) + conditional_adjustment
```

**Process:** Updates ratings of similar dishes based on user feedback with weighted adjustments

## Technical Architecture

```
├── Data Layer
│   ├── Food survey.csv (User ratings)
│   └── temp_dish_inventory.csv (Dish information)
│
├── Algorithm Layer
│   ├── Cosine Similarity (Dish similarity calculation)
│   ├── Linear Regression (New user prediction)
│   └── Recommendation Filtering Logic
│
└── User Interface Layer
    ├── User Authentication
    ├── Ingredient Selection
    ├── Meal Time Selection
    ├── Recommendation Display
    └── Feedback Collection
```

## Installation & Usage

### Prerequisites

- Python 3.7+
- pandas
- numpy
- scikit-learn

### Setup

1. Clone the repository

   ```
   git clone https://github.com/yourusername/meal-harmony.git
   cd meal-harmony
   ```

2. Install required dependencies

   ```
   pip install pandas numpy scikit-learn
   ```

3. Ensure data files are in the correct location:

   - `Food survey.csv`
   - `temp_dish_inventory.csv`

4. Run the application
   ```
   python food_recommendation.py
   ```

## Future Integrations

### 1. Nutritional Intelligence

- Integration of caloric and nutritional data
- Dietary restriction handling (vegan, gluten-free, etc.)
- Meal balancing for optimal nutrition across multiple days

### 2. Environmental Context Integration

- Weather-based recommendations (hot/cold dishes appropriate for the season)
- Time-of-day adjustments (lighter meals in evening, energy-rich in morning)
- Regional ingredient availability considerations

### 3. Advanced User Modeling

- Multi-user household preference balancing
- Preference drift detection over time
- Taste profile clustering for improved recommendations

### 4. Preparation Complexity Analysis

- Time and effort estimation for meal preparation
- Kitchen equipment requirements assessment
- Cooking skill level consideration

### 5. Smart Grocery Integration

- Automated shopping list generation
- Ingredient substitution recommendations
- Waste reduction through inventory management

## Team Information

This project was developed as a Minor Project in the domain of Machine Learning & Statistics.

## License

[MIT License](LICENSE)

Copyright (c) 2025 Meal HARMONY Team
