
from typing import List
import gradio as gr
from music_search.search import NearestNeighbor
from music_search.util import get_config


config = get_config("configs/default.yaml")
nn = NearestNeighbor(config.index_path, config.db_path, read_index=True)
k = 8

def search(album_description: str) -> List[str]:
    records = nn.search(album_description, k=8)
    output = []
    for i, record in enumerate(records):
        print(record)
        score, index_id, sent_id, doc_id, review, title = record.values()
        markdown = (
            f"### {i+1} - {title}\n"
            f"score: {score:.2f} - iid: {index_id} - sent_id: {sent_id} - doc_id: {doc_id}\n"
            "\n"
            f"> {review}"
        )
        output.append(markdown)

    return output

def main():
    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # Album similarity search
        Type an album description to search for similar albums based on pitchfork reviews!
        """)
        with gr.Row():
            with gr.Column(scale=1, min_width=500):
                desc = gr.Textbox(label="Album description", lines=5)
                search_btn = gr.Button("Search")
            with gr.Column(scale=2, min_witdh=800):
                gr.Markdown("## Results")
                outputs = [gr.Markdown(label=f"Hit {i}") for i in range(1, k+1)]
                search_btn.click(fn=search, inputs=desc, outputs=outputs)

    demo.launch()
    

if __name__ == '__main__':
    main()

