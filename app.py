import gradio as gr
import json
import threading
from datetime import datetime
import os
from typing import Dict, Any

# Import your model functions directly instead of making HTTP calls
from model import (
    load_or_train_model,
    predict_user_input,
    retrain_model_with_feedback,
    store_predictions,
    fetch_feedback_for_user,
)

# Load model once at startup
print("Loading model...")
model = load_or_train_model()
print("Model loaded successfully!")

def predict_consumption(adult_male, adult_female, child, region, season, event, stocks_text):
    try:
        # Parse the stocks - expecting format like: rice:5, milk:3, potato:2
        stocks = {}
        for item in stocks_text.split(','):
            if ':' in item:
                product, amount = item.strip().split(':')
                stocks[product.strip()] = float(amount.strip())
        
        # Prepare the input (same structure as before)
        user_input = {
            "family": {
                "adult_male": int(adult_male),
                "adult_female": int(adult_female),
                "child": int(child)
            },
            "region": region,
            "season": season,
            "event": event,
            "stock": stocks,
            "user_id": f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Call the prediction function directly instead of making HTTP request
        results = predict_user_input(user_input, model)
        
        # Store predictions (same as before)
        store_predictions(user_input["user_id"], results, user_input)
        
        # Format results for display
        formatted_results = []
        for product, prediction in results.items():
            formatted_results.append(f"""
🥘 **{product.upper()}:**
📊 Daily Consumption: {prediction['predicted_consumption']} kg
⏰ Will finish in: {prediction['predicted_finish_days']} days
📅 Finish Date: {prediction['predicted_finish_date']}
⚠️ Prediction Error: ±{prediction['predicted_finish_error']} days
""")
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nMake sure to format stocks as: rice:5, milk:3, potato:2"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="🛒 Grocy Genie") as iface:
    gr.Markdown("# 🛒 Grocy Genie - Smart Grocery Consumption Predictor")
    gr.Markdown("Predict how long your groceries will last based on family size, region, season, and events.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👨‍👩‍👧‍👦 Family Information")
            adult_male = gr.Number(label="Adult Males", value=2, minimum=0, maximum=10)
            adult_female = gr.Number(label="Adult Females", value=2, minimum=0, maximum=10)
            child = gr.Number(label="Children", value=1, minimum=0, maximum=10)
            
            gr.Markdown("### 🏠 Context Information")
            region = gr.Dropdown(choices=["urban", "rural"], label="Region", value="urban")
            season = gr.Dropdown(choices=["winter", "spring", "summer", "autumn"], label="Season", value="summer")
            event = gr.Dropdown(choices=["normal", "fasting", "guests", "sickness", "travel", "meal_off"], label="Event", value="normal")
            
            gr.Markdown("### 🛍️ Current Stock")
            stocks_text = gr.Textbox(
                label="Stock (format: product:amount, product:amount)", 
                value="rice:5, milk:3, potato:2",
                placeholder="rice:5, milk:3, potato:2, onion:1",
                lines=2
            )
        
        with gr.Column():
            gr.Markdown("### 📊 Predictions")
            output = gr.Textbox(label="Results", lines=15, max_lines=20)
            predict_btn = gr.Button("🔮 Predict Consumption", variant="primary", size="lg")
    
    predict_btn.click(
        predict_consumption,
        inputs=[adult_male, adult_female, child, region, season, event, stocks_text],
        outputs=output
    )
    
    gr.Markdown("### 📝 Example Usage")
    gr.Markdown("""
    - **Family**: 2 adult males, 2 adult females, 1 child
    - **Context**: Urban area, summer season, normal event
    - **Stock**: rice:5, milk:3, potato:2, onion:1
    
    The model will predict daily consumption and when each item will run out.
    """)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)