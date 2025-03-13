import pandas as pd
import ollama
import gradio as gr
import matplotlib.pyplot as plot
import io
from PIL import Image
import itertools

def getCSVFile(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna("") 
    formatted_text = "\n".join([", ".join(f"{col}: {row[col]}" for col in df.columns) for _, row in df.iterrows()])
    return df, formatted_text

def UploadFile(file):
    UploadFileForGraph(file)
    global csvData, dataframe_qa
    dataframe_qa, csvData = getCSVFile(file) 
    return "QA File uploaded successfully! You can now ask questions."

def UploadFileForGraph(file):
    global dataframe_graph
    dataframe_graph = pd.read_csv(file)
    return "Graph File uploaded successfully! You can now generate graphs."

def GetResponse(model_name, context, question):
    prompt = f"""
    COnsider that you are an AI assistant and have to analyze the following dataset:
    {context}
    Then
    Answer the following question based on the above dataset:
    {question}
    """
    response = ollama.chat(model=model_name, messages=[{"role": "system", "content": "consider yourself as an expert data analyst."},
                                                        {"role": "user", "content": prompt}],
                            stream=False)
    return response['message']['content']


def GradioInterface(model_name, question):
    global csvData
    if not csvData:
        return "Please upload a CSV file first."
    return GetResponse(model_name, csvData, question)

def GraphGeneration():
    global dataframe_graph
    if dataframe_graph is None:
        return "Please upload a CSV file first for graph generation."
    
    plots = []
    numerical_columns = dataframe_graph.select_dtypes(include=['number']).columns
    if len(numerical_columns) < 2:
        return "At least two numerical columns are required for graph generation."
    
    for col_x, col_y in itertools.combinations(numerical_columns, 2):
        fig, ax = plot.subplots()
        ax.plot(dataframe_graph[col_x], dataframe_graph[col_y], marker='o', linestyle='-')  # Line plot instead of scatter
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f'Line Plot: {col_x} vs {col_y}')
        
        buf = io.BytesIO()
        plot.savefig(buf, format='png')
        plot.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        plots.append(img)
    
    return plots

def get_local_models():
    try:
        models = ollama.list()
        return [model.model for model in models.models]
    except Exception as e:
        print("Error fetching models:", e)
        return ["Error fetching models"]

def fetch_local_models():
    try:
        models = ollama.list().models
        return [model.model for model in models]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["Error fetching models"]

uploaded_file=None
csvData = ""
dataframe_qa = None
dataframe_graph = None

demo = gr.Blocks(theme=gr.themes.Soft())
with demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload CSV File")
            model_dropdown = gr.Dropdown(get_local_models(), label="Select Model",value="llama3.1:latest")
            file_upload = gr.File(label="Upload CSV", type="filepath")
            upload_button = gr.Button("Upload")
            progress_bar = gr.Markdown("")
            upload_button.click(UploadFile, inputs=file_upload, outputs=progress_bar)
        
        with gr.Column():
            gr.Markdown("### Ask Queries")
            chatbot_input = gr.Textbox(label="Ask a question", lines=2)
            submit_button = gr.Button("Submit")
            output_box = gr.Textbox(label="Chat Output", interactive=True, lines=9)
            submit_button.click(GradioInterface, inputs=[model_dropdown, chatbot_input], outputs=output_box)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Generate Graphs")
            generate_button = gr.Button("Generate Graphs")
            graph_output = gr.Gallery()
            generate_button.click(GraphGeneration, outputs=graph_output)

demo.launch()
