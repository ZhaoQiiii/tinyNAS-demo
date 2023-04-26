import gradio as gr
from classification import get_classification_interface
from detection import get_detection_interface
from clip_cn import get_clip_cn_interface

if __name__ == "__main__":
    gr.close_all()
    with gr.TabbedInterface(
            [get_classification_interface(), get_detection_interface(), get_clip_cn_interface() ],
            ["分类", "检测", "中文CLIP"],
        ) as demo:

        #demo.queue(concurrency_count=1)
        demo.launch()
