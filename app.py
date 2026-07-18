import gradio as gr
import joblib

# Load trained model
deployed_lr = joblib.load("myFirstModel.pkl")

# Prediction function
def predict_rent(size_of_prop):
    prediction = deployed_lr.predict([[size_of_prop]])
    return f"🏠 Estimated Monthly Rent: ₹ {prediction[0]:,.2f}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_rent,

    inputs=gr.Number(
        label="🏡 Property Size (sq.ft)"
    ),

    outputs=gr.Textbox(
        label="💰 Predicted Rent"
    ),

    title="🏠 Property Rent Predictor",

    description="""
Predict property rent instantly using a Machine Learning model.
Enter the property size below to get an estimated monthly rent.
""",

    theme=gr.themes.Soft()
)

# Launch app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
