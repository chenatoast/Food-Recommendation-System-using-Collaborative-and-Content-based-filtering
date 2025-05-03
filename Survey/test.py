import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("Food survey.csv")

# Extract dish ratings
dishes = df.iloc[:, 1:].copy()
dish_names = dishes.columns.values

# Simulated user IDs
users = {i: str(i) for i in range(36)}

# Calculate dish similarity
dish_similarity = cosine_similarity(dishes.T)

def item_similarity(interaction_matrix):
    item_interaction_matrix = interaction_matrix.T
    similarity_matrix = cosine_similarity(item_interaction_matrix)
    return similarity_matrix

def select_neighborhood(similarity_matrix, item_id, neighborhood_size):
    item_similarity_scores = similarity_matrix[item_id]
    sorted_indices = np.argsort(item_similarity_scores)[::-1]
    neighborhood = sorted_indices[1:neighborhood_size+1]  # Exclude the item itself
    return neighborhood

def update_data(user_id, selected_dish, neighborhood_size=5):
    hood = select_neighborhood(dish_similarity, selected_dish, neighborhood_size)

    for i in hood:
        adjustment = 0.5 if dish_similarity[selected_dish][i] < 0.5 else -0.7
        dish_similarity[selected_dish][i] += adjustment
        dishes.iloc[user_id][i] = np.clip(dishes.iloc[user_id][i] + adjustment, 1, 5)

# Train a linear regression model to predict ratings for new users
x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0]).reshape(-1, 1)
model = LinearRegression().fit(y, x)

def add_user(user_id, name):
    new_data = model.predict([[user_id]])
    dishes.loc[len(dishes)] = np.clip(new_data[0], 1, 5)
    users[user_id] = name

def validate_user(user_id):
    return user_id in dishes.index

def get_recommendations(user_id, num_recommendations=5):
    user_ratings = dishes.iloc[user_id].values
    unrated_dishes = [(i, user_ratings[i]) for i in range(len(user_ratings)) if user_ratings[i] < 3]
    sorted_unrated = sorted(unrated_dishes, key=lambda x: x[1], reverse=True)
    return sorted_unrated[:num_recommendations]

def interact(user_id):
    print(f"Welcome, {users[user_id]}")
    while True:
        print("\nEnter your choice, enter 99 to exit, or 111 for recommendations.")
        selection = int(input())

        if selection == 99:
            break
        elif selection == 111:
            recommendations = get_recommendations(user_id)
            print("You might enjoy:")
            for i, _ in recommendations:
                print(f"{i}: {dish_names[i]}")
            selection = int(input("Enter your selection: "))
            update_data(user_id, selection)
        else:
            print("\nGreat choice, enjoy your meal.")
            update_data(user_id, selection)

# Main loop
while True:
    user_id = int(input("Enter your ID: "))
    if not validate_user(user_id):
        print("User doesn't exist, creating account...")
        name = input("Please enter your username: ")
        add_user(user_id, name)
    else:
        interact(user_id)
