import os
import uuid
import gradio as gr
from gradio import components
from fastmcp import FastMCP
# from core.parser import parse_document, parse_url
from core.parser import parse_document, parse_url
from core.summarizer import summarize_content, tag_content
from core.storage import add_document, search_documents
from core.agent import answer_question
# from core.components import DocumentViewer
import plotly.graph_objects as go

# Initialize the FastMCP server (for agentic tools)
mcp = FastMCP("IntelligentContentOrganizer")

# Gradio UI functions
def process_content(file_obj, url, tags_input):
    """
    Handle file upload or URL input: parse content, summarize, tag, store.
    """
    content_text = ""
    source = ""
    if file_obj is not None:
        # Save uploaded file to temp path
        file_path = file_obj.name
        content_text = parse_document(file_path)
        source = file_obj.name
    elif url:
        content_text = parse_url(url)
        source = url
    else:
        return "No document provided.", "", "", ""

    # Summarize and tag (simulated)
    summary = summarize_content(content_text)
    tags = tag_content(content_text)

    # Allow user to override or confirm tags via input
    if tags_input:
        # If user entered new tags, split by comma
        tags = [t.strip() for t in tags_input.split(",") if t.strip() != ""]

    # Store in ChromaDB with a unique ID
    doc_id = str(uuid.uuid4())
    metadata = {"source": source, "tags": tags}
    add_document(doc_id, content_text, metadata)

    return content_text, summary, ", ".join(tags), f"Document stored with ID: {doc_id}"

def generate_graph():
    """
    Create a simple Plotly graph of documents.
    Nodes = documents, edges = shared tags.
    """
    # Fetch all documents from ChromaDB
    from core.storage import get_all_documents
    docs = get_all_documents()
    if not docs:
        return go.Figure()  # empty

    # Build graph connections: if two docs share a tag, connect them
    nodes = {doc["id"]: doc for doc in docs}
    edges = []
    for i, doc1 in enumerate(docs):
        for doc2 in docs[i+1:]:
            shared_tags = set(doc1["metadata"]["tags"]) & set(doc2["metadata"]["tags"])
            if shared_tags:
                edges.append((doc1["id"], doc2["id"]))

    # Use networkx to compute layout (or simple fixed positions)
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)

    # Create Plotly traces
    edge_x = []
    edge_y = []
    for (src, dst) in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        text = nodes[node_id]["metadata"].get("source", "")
        node_text.append(f"{text}\nTags: {nodes[node_id]['metadata']['tags']}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=10, color='skyblue'),
        text=node_text, hoverinfo='text', textposition="bottom center")

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title="Document Knowledge Graph",
                                     showlegend=False,
                                     margin=dict(l=20, r=20, b=20, t=30)))
    return fig

def handle_query(question):
    """
    Answer a user question by retrieving relevant documents and summarizing them.
    """
    if not question:
        return "Please enter a question."

    answer = answer_question(question)
    return answer

# Build Gradio interface with Blocks
with gr.Blocks(title="Intelligent Content Organizer") as demo:
    gr.Markdown("# Intelligent Content Organizer")
    with gr.Tab("Upload / Fetch Content"):
        gr.Markdown("**Add a document:** Upload a file or enter a URL.")
        with gr.Row():
            file_in = gr.File(label="Upload Document (PDF, TXT, etc.)")
            url_in = gr.Textbox(label="Document URL", placeholder="https://example.com/article")
        tags_in = gr.Textbox(label="Tags (comma-separated)", placeholder="Enter tags or leave blank")
        process_btn = gr.Button("Parse & Add Document")
        doc_view = gr.Textbox(label="Document Preview", lines=10, interactive=False)
        summary_out = gr.Textbox(label="Summary", interactive=False)
        tags_out = gr.Textbox(label="Detected Tags", interactive=False)
        status_out = gr.Textbox(label="Status/Info", interactive=False)
        process_btn.click(fn=process_content, inputs=[file_in, url_in, tags_in],
                          outputs=[doc_view, summary_out, tags_out, status_out])

    with gr.Tab("Knowledge Graph"):
        gr.Markdown("**Document relationships:** Shared tags indicate edges.")
        graph_plot = gr.Plot(label="Knowledge Graph")
        refresh_btn = gr.Button("Refresh Graph")
        refresh_btn.click(fn=generate_graph, inputs=None, outputs=graph_plot)

    with gr.Tab("Ask a Question"):
        gr.Markdown("**AI Q&A:** Ask a question about your documents.")
        question_in = gr.Textbox(label="Your Question")
        answer_out = gr.Textbox(label="Answer", interactive=False)
        ask_btn = gr.Button("Get Answer")
        ask_btn.click(fn=handle_query, inputs=question_in, outputs=answer_out)

if __name__ == "__main__":
    # Launch Gradio app (Hugging Face Spaces will auto-launch this)
    # demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
    demo.launch(mcp_server=True)
