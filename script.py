import pandas as pd
import itertools

# Define state categories
battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
temperatures = ["Safe", "Warning", "Critical"]
threat_states = ["Normal", "Confirming", "Confirmed"]

# Define decision rules (Example based on previous analysis)
# Action codes: 0 = No DDoS, 1 = XGBoost, 2 = TST

def decision_rule(battery, temp, threat):
    # Critical system protection
    if battery == "0-20%" or temp == "Critical":
        return 0

    if threat == "Normal":
        if battery in ["0-20%", "21-40%"]:
            return 0
        else:
            return 1

    if threat == "Confirming":
        if battery in ["0-20%", "21-40%"]:
            return 1
        else:
            if temp == "Warning" or temp == "Safe":
                return 2
            else:
                return 1

    if threat == "Confirmed":
        if battery in ["0-20%", "21-40%"]:
            return 1
        else:
            return 1

# Generate all combinations
all_states = list(itertools.product(battery_levels, temperatures, threat_states))

# Apply decision rule to each state
lookup_table = []
for battery, temp, threat in all_states:
    action = decision_rule(battery, temp, threat)
    lookup_table.append({
        "Battery_Level": battery,
        "Temperature": temp,
        "Threat_State": threat,
        "Action": action
    })

# Create DataFrame
df_lookup = pd.DataFrame(lookup_table)

# Map action to labels
action_labels = {0: "No DDoS", 1: "XGBoost", 2: "TST"}
df_lookup["Action_Label"] = df_lookup["Action"].map(action_labels)

# Save to Excel file
excel_filename = "ddos_rl_lookup_table.xlsx"
df_lookup.to_excel(excel_filename, index=False)

excel_filename, df_lookup.head()