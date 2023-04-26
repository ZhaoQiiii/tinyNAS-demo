import gradio as gr
from common import *

markdown = r'''
### 使用方法

1. 资源设定网络

    用户根据自己的场景需要，将诸如网络参数量，计算量等资源限制告知系统，TinyNAS 就会搜索满足用户要求的网络。在本 demo 中，用户只需要通过拖动条进行设置即可，具体参数的定义解释如下表所示：

    | 序号 | 约束| 定义解释 | 
    | --- | --- | --- | 
    | 1	| Max Params（M) | 搜索出来的各种网络结构中，其参数量最大不能超过多少（参考值：30-80M） | 
    | 2	| Max FLOPs（M)	| 搜索出来的各种网络结构中，其浮点运算次数最大不能超过多少（参考值：6.3G-20.3G） | 
    | 3	| Iter Num | 搜索网络结构时搜索的迭代次数，可以认为是搜索时长的另一种体现（参考值：1000-2000） | 

2. 执行搜索

    在资源设定完成后，用户只需要点击提交，系统就会执行搜索，完毕后在右侧就会返回搜索到的网络结构及其使用示例，用户下载即可使用。

3. 下载搜索结果 tar 包之后，用户执行以下命令，即可快速体验

    <details>
    <summary>查看代码</summary>


    ```bash
    tar xzvf xxx.tar.gz
    cd 解压目录
    python3 demo.py
    ```

    </details>

    - demo.py包含了完整的模型加载、数据加载和预处理、模型推理等步骤，能够帮助您快速集成中文 CLIP 模型到开发项目中
    - 该运行示例会从 MS COCO 数据集下载一张图片（如下所示），并预测其与两个短句 '一张猫的照片' 和 '一张狗的照片' 的匹配度；输出结果为 [0.99, 0.01]

        <img src="https://vcs-dockers.oss-cn-hangzhou.aliyuncs.com/TinyNas/misc/clip_cn/000000039769.jpeg" width="300">

### 使用场景
- 通用的图文跨模态检索任务
- 通用图文特征提取器
'''

def get_clip_cn_interface():
    def run_func( budget_model_size, budget_flops, ea_num_random_nets ,num_network = 1):
        budget_model_size, budget_flops = adjust_data_range(budget_model_size, budget_flops)
        params = locals()
        params["space_arch"] = "CLIP_CN"
        
        url = client.commit(params)
        if url is None:
            url = 'https://vcs-dockers.oss-cn-hangzhou.aliyuncs.com/TinyNas%2Ff18565d8-5876-11ed-9c76-00163e098124.tar.gz?OSSAccessKeyId=LTAI4GDnHJDwQebpBxVbdEUf&Expires=1982584733&Signature=nlRv63AqaN8jseJDIUUIR1CZ%2Bjc%3D'
            logger.info( 'use demo package')
        logger.info( 'get reponse url {} '.format( url))   
        
        filename = url.split('?')[0].split('%')[-1].split('%')[-1][2:]
        linkWrapper = f"""
          <div align="center">
            <div class="gr-button gr-button-lg gr-button-primary">
            <a href="{url}"">下载模型</a>
            </div>
          </div>
        """
        return [url, linkWrapper]

    description = '中文 CLIP 模型在约2亿图文对的大规模中文数据上训练，每次训练的计算代价和时间开销极其昂贵。使用我们提供的中文 CLIP NAS 服务，几分钟内便可获得满足定制化需求的预训练中文 CLIP 模型。针对中文 CLIP 多模态大模型的预训练特点，我们依托达摩院自研的 NAS 算法，设计基于权重的 zero-cost score，对 Transformer 结构的性能进行快速评估，结合权重共享的 one-shot 训练方法，得到一次训练多次部署的 once-for-all supernet。'
    
    budget_model_size = gr.Slider(30, 80, step= 0.01, value=65,label="Max Params (M)")
    budget_flops = gr.Slider(6.3e3, 20.3e3, step = 10, value=6.3e3, label="Max FLOPs (M)" )
    ea_num_random_nets = gr.Slider(1000, 2000, step=1 , value=1000, label="Iter Num")
    num_network= gr.Slider(1, 5, step=1, value=1, label="How many networks do you want?")
    output_url = gr.TextArea(label="下载链接")
    defaultBtn = """
        <div align="center">
            <div class="gr-button gr-button-lg gr-button-primary" style="background-color: #ddd;">
            下载模型
            </div>
        </div>
    """

    inputs = [ budget_model_size, budget_flops, ea_num_random_nets] # ,num_network ]
    outputs = [output_url, gr.HTML(value = defaultBtn)]

    interface = gr.Interface(
                fn = run_func,
                inputs = inputs,
                outputs = outputs,
                title = title,
                description = description,
                article = (markdown),
                allow_flagging='never')
    return interface

if __name__ == "__main__":
    gr.close_all()
    with gr.TabbedInterface(
            [get_clip_cn_interface()],
            ["中文CLIP"],
        ) as demo:

        demo.queue(concurrency_count=1)
        demo.launch()
