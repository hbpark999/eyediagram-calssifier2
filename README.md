# Eye Diagram Analyzer

This project provides a Streamlit-based web application for analyzing eye diagrams. It includes mask evaluation, classification using a CNN model, and integration with GPT for detailed analysis.

## Streamlit Sharing Deployment

This app is deployed on Streamlit Sharing. You can access it [here](your_streamlit_sharing_url).

## Local Setup (for development)

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/eye-diagram-analyzer.git
   cd eye-diagram-analyzer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - For local development, create a `.env` file in the project root and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```
   - For Streamlit Sharing, set this as a secret in the Streamlit Sharing dashboard.

4. Ensure you have the trained model file `eye_diagram_classifier_v1.pth` in the `models/` directory.

## Running the Application Locally

To run the Streamlit app locally:

```
streamlit run app.py
```

Then, open your web browser and go to the URL provided by Streamlit (usually http://localhost:8501).

## Usage

1. Upload an eye diagram image.
2. Adjust the mask parameters in the sidebar if needed.
3. The app will perform mask evaluation and display the result.
4. If the mask evaluation result is NG, the app will classify the eye diagram and provide GPT analysis.

## Streamlit Sharing Deployment

The app is automatically deployed to Streamlit Sharing when changes are pushed to the main branch of the GitHub repository. Ensure that you've set up the following in Streamlit Sharing:

1. Connect your GitHub repository to Streamlit Sharing.
2. Set the OpenAI API key as a secret in the Streamlit Sharing dashboard.
3. Ensure the `requirements.txt` file is up to date with all necessary dependencies.

## Note

This application requires a trained PyTorch model for eye diagram classification. Ensure you have the correct model file in the `models/` directory before running the application or deploying to Streamlit Sharing.