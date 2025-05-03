import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

#! Load data
df = pd.read_csv("Food survey.csv")

#! Set UserID as index and extract dish ratings
df.set_index('UserID', inplace=True)
dishes = df.copy()
dish_names = dishes.columns.values

#! Simulate user IDs from the CSV file
users = {user_id: str(user_id) for user_id in df.index}

#! Calculate dish similarity
dish_similarity = cosine_similarity(dishes.T)
print(dish_similarity.shape)

inventory_df = pd.read_csv('temp_dish_inventory.csv')

ingredient_columns = inventory_df.columns[2:-4].tolist()
meal_time_columns = inventory_df.columns[-4:].tolist()

ingredients = inventory_df[ingredient_columns].values
meal_times = inventory_df[meal_time_columns].values
 
#! Dictionary to track recently selected dishes for each user
recently_selected = {}

user_selected_ingredients = []
user_meal_time = []


#! Function to ask the user for their meal time preference
def ask_meal_time():
    meal_time_map = {
        1: 'Breakfast',
        2: 'Lunch',
        3: 'Dinner',
        4: 'Snacks'
    }

    print("\nFor which meal time would you like recommendations?")

    #? Display meal time options using meal_time_map
    for num, time in meal_time_map.items():
        print(f"{num}: {time}")

    selected_time = int(input("\nEnter the number corresponding to your meal time: "))

    if selected_time in meal_time_map:
        selected_meal_time = meal_time_map[selected_time]
        print(f"You have selected: {selected_meal_time}")
    else:
        print("Invalid selection, defaulting to Lunch")
        selected_meal_time = 'Lunch'

    return {selected_meal_time}


#! Function to select the neighborhood of similar items for a given item
def select_neighborhood(similarity_matrix, item_id, neighborhood_size):
    item_similarity_scores = similarity_matrix[item_id]
    sorted_indices = np.argsort(item_similarity_scores)[::-1]
    neighborhood = sorted_indices[1:neighborhood_size+1]  # Exclude the item itself
    return neighborhood


#! Function to get the user's rating for the recommendations
def get_recommendation_rating():
    while True:
        try:
            rating = int(input("\nRate the recommendation quality (1 to 5): "))
            if 1 <= rating <= 5:
                print(f"Thank you! You rated the recommendation as: {rating}/5.")
                return rating
            else:
                print("Please enter a rating between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter an integer between 1 and 5.")


#! Function to update the data based on the user's selection and adjust the ratings of similar dishes
def update_data(user_id, selected_dish, recommendation_rating, neighborhood_size=5):
    hood = select_neighborhood(dish_similarity, selected_dish, neighborhood_size)

    rating_adjustment = 0.1 * (recommendation_rating - 3)

    for i in hood:
        current_rating = dishes.loc[user_id, dish_names[i]]
        if dish_similarity[selected_dish][i] < 0.5:
            adjustment = 0.1
        else:
            adjustment = -0.2 if current_rating > 1.2 else 0

        final_adjustment = rating_adjustment + adjustment

        new_rating = np.clip(current_rating + final_adjustment, 1, 5)
        new_rating = round(new_rating, 1)
        # print(f"User {user_id}: Dish '{dish_names[i]}' - Previous Rating: {current_rating}, Updated Rating: {new_rating}")
        dishes.loc[user_id, dish_names[i]] = new_rating

    #? Track the recently selected dish
    if user_id not in recently_selected:
        recently_selected[user_id] = []
    recently_selected[user_id].append(selected_dish)

    #? Limit the number of tracked recent dishes (e.g., 3)
    if len(recently_selected[user_id]) > 3:
        recently_selected[user_id].pop(0)

    #? Save the updated data to CSV
    df.loc[user_id] = dishes.loc[user_id]
    df.to_csv("Food survey.csv")


#? Train a linear regression model to predict ratings for new users
x = np.array(df)
y = np.arange(len(df)).reshape(-1, 1)
linear_model = LinearRegression().fit(y, x)


#! Function to add a new user to the system and predict ratings for them using the linear model
def add_user(user_id, name):
    global dishes

    new_data = linear_model.predict([[user_id]])
    new_ratings = np.clip(new_data[0], 1, 5)
    new_ratings = np.round(new_ratings, 1)

    new_user_data = pd.Series(new_ratings, index=dishes.columns, name=user_id)
    dishes = pd.concat([dishes, new_user_data.to_frame().T])

    df.loc[user_id] = new_ratings
    df.to_csv("Food survey.csv")

    users[user_id] = name

    print(f"Account created successfully! Welcome, {name}.\n")


#! Function to validate if a user exists in the system based on their ID
def validate_user(user_id):
    return user_id in dishes.index


#! Function to get recommendations for a user based on their ratings and selected ingredients and meal time preferences
def get_recommendations(user_id, num_recommendations=5):
    global user_selected_ingredients, user_meal_time

    user_ratings = dishes.loc[user_id].values
    unrated_dishes = [(i, user_ratings[i]) for i in range(len(user_ratings)) if user_ratings[i] < 3]
    sorted_unrated = sorted(unrated_dishes, key=lambda x: x[1], reverse=True)

    recommendations = [(i, score) for i, score in sorted_unrated if i not in recently_selected.get(user_id, [])]

    #? Filter based on ingredients and retry with cosine similarity
    filtered_recommendations = []
    for i, score in recommendations:
        if check_ingredients(i, user_selected_ingredients):
            filtered_recommendations.append((i, score))

    if not filtered_recommendations:
        print("\nNo dishes fully match your available ingredients and selected meal time.")
        print("Retrying with more cosine similarity recommendations...\n")
        filtered_recommendations = retry_cosine_similarity(user_id, recommendations)

    return filtered_recommendations[:num_recommendations] if filtered_recommendations else []


#! Function to check if a dish's ingredients are available to the user based on their selection
def check_ingredients(dish_id, available_ingredients):
    dish_row = inventory_df[inventory_df['Item_id'] == dish_id]
    if dish_row.empty:
        return False

    dish_ingredients = set(
        ingredient_columns[j]
        for j in range(len(ingredient_columns))
        if dish_row[ingredient_columns[j]].values[0] == 1
    )

    available_ingredients_set = set(available_ingredients)
    return dish_ingredients.issubset(available_ingredients_set)


#! Function to retry recommendations using cosine similarity for dishes that have similar ingredients to the user's selection but were not recommended initially due to low ratings
def retry_cosine_similarity(user_id, recommendations):
    retry_recommendations = []
    for i, score in recommendations:
        hood = select_neighborhood(dish_similarity, i, 5)
        for neighbor in hood:
            if check_ingredients(neighbor, user_selected_ingredients):
                retry_recommendations.append((neighbor, dish_similarity[i][neighbor]))

    return retry_recommendations if retry_recommendations else []


#! Function to prompt the user to select a dish from the recommended list
def select_from_recommendations(user_id, recommendations):
    print("\nPlease select a dish from the recommended list:")
    for idx, (dish_id, _) in enumerate(recommendations):
        print(f"{idx + 1}: {dish_names[dish_id]}")
    choice = int(input("\nEnter the number of your selected dish: ")) - 1

    if 0 <= choice < len(recommendations):
        selected_dish = recommendations[choice][0]
        print(f"You selected: {dish_names[selected_dish]}")
        return selected_dish
    else:
        print("Invalid selection.")
        return None

def interact(user_id):
    global user_selected_ingredients, user_meal_time
    print(f"Welcome, {users[user_id]}")

    if not user_selected_ingredients:
        print("Please select the ingredients you have available:")

        for i in range(0, len(ingredient_columns), 6):
            row = ingredient_columns[i:i+6]
            print("\t\t".join([f"{i+j+1}: {ingredient}" for j, ingredient in enumerate(row)]))

        selected_indices = input("\nEnter the numbers of ingredients you have (comma separated): ")
        selected_indices = [int(i) - 1 for i in selected_indices.split(",")]  # convert input to indices

        user_selected_ingredients = [ingredient_columns[i] for i in selected_indices]  # Store user selection
        print(f"\nYou have selected: {', '.join(user_selected_ingredients)}")

    while True:
        print("\nWhat would you like to do?")
        print("1: Get Recommendations")
        print("2: View Recently Selected Dishes")
        print("99: Exit\n")

        choice = input("Enter your choice: ")

        if choice == "99":
            print("Thank you for using the Food Recommendation System. Goodbye!")
            break
        elif choice == "1":
            user_meal_time = ask_meal_time()
            recommendations = get_recommendations(user_id)

            if recommendations:
                print("\nWe recommend the following dishes:")
                for i, _ in recommendations:
                    print(f"{i}: {dish_names[i]}")

                #? Ask user to rate the entire list of recommendations
                combined_rating = get_recommendation_rating()

                #? Update ratings based on the combined score for all recommended dishes
                for i, _ in recommendations:
                    update_data(user_id, i, combined_rating)

                #? Ask user to select a dish from the recommendations
                selected_dish = select_from_recommendations(user_id, recommendations)
                if selected_dish is not None:
                    update_data(user_id, selected_dish, combined_rating)

                print("\nYour feedback has been recorded. Enjoy your meal!")
            else:
                print("No suitable recommendations found. Try again with different ingredients or meal time.")
        # elif choice == "2":
        #     print("Please enter the number of the dish you want to select:")
        #     for i, dish in enumerate(dish_names):
        #         print(f"{i}: {dish}")

        #     selection = int(input("\nEnter the number of your selected dish: "))
        #     update_data(user_id, selection)

            print("Your selection has been saved. Enjoy your meal!")
        elif choice == "2":
            print("Your recently selected dishes are:")
            for dish_id in recently_selected.get(user_id, []):
                print(dish_names[dish_id])
        else:
            print("Invalid choice. Please try again.")


while True:
    user_id = int(input("Enter your ID: "))
    if not validate_user(user_id):
        print("User doesn't exist, creating account...")
        name = input("\nPlease enter your username: ")
        add_user(user_id, name)
    else:
        interact(user_id)
        break
