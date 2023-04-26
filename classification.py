import gradio as gr
from common import *

markdown = r'''
### 使用方法

1. 资源设定网络

    用户根据自己的场景需要，将诸如网络参数量，计算量等资源限制告知系统，TinyNAS 就会搜索满足用户要求的网络。在本 demo 中，用户只需要通过拖动条进行设置即可，具体参数的定义解释如下表所示：

    | 序号 | 约束| 定义解释 | 
    | --- | --- | --- | 
    | 1	| Class Num	| 分类任务对应的类别数量 | 
    | 2	| Max Params（M) | 搜索出来的各种网络结构中，其参数量最大不能超过多少（参考值：10-50M） | 
    | 3	| Max FLOPs（M)	| 搜索出来的各种网络结构中，其浮点运算次数最大不能超过多少（参考值：2-8G） | 
    | 4	| Max Layers | 搜索出来的各种网络结构中，其网络层数（也表示网络深度）最大不能超过多少（参考值：16-200） | 
    | 5	| Iter Num | 搜索网络结构时搜索的迭代次数，可以认为是搜索时长的另一种体现（参考值：1000-2000） | 

2. 执行搜索

    在资源设定完成后，用户只需要点击提交，系统就会执行搜索，完毕后在右侧就会返回搜索到的网络结构及其使用示例，用户下载即可使用。

3. 下载搜索结果tar包之后，用户执行以下命令，即可对搜索得到的网络结构进行一次前向推理
    
    <details>
    <summary>查看代码</summary>

    ```bash
    tar xzvf xxx.tar.gz
    cd 解压目录
    python3 demo.py
    ```

    </details>

    用户可以基于demo.py中的示例代码方便地将 NAS 网络集成到训练流程中。

### 一个例子：TinyNAS 与 Timm 集成
1. 首先按照 Timm 的<a href="https://github.com/rwightman/pytorch-image-models/">官方代码库</a>对其环境依赖进行安装。 克隆的 Timm 代码库命名为 pytorch-image-models
2. 将搜索结果解压到 Timm 所在目录中的

    <details>
    <summary>查看代码</summary>


    ```bash
    cp -r <解压目录>/* pytorch-image-models/
    ```

    </details>

3. 将 TinyNAS 网络结构实例化，修改 pytorch-image-models/train.py

    <details>
    <summary>查看代码</summary>

    ```python
    import ast
    # 需要在 models 前添加 . 表示从当前目录导入
    from .models import __all_masternet__
    ################################
    #                              #
    #     加载搜索到的 backbone      #
    #                              #
    ################################ 
    def get_backbone(filename,
                    load_weight=True,
                    network_id=0,
                    task='classification'):
        # load best structures
        with open(filename, 'r') as fin:
            content = fin.read()
            output_structures = ast.literal_eval(content)
        network_arch = output_structures['space_arch']
        best_structures = output_structures['best_structures']
        out_indices = (1, 2, 3, 4) if task == 'detection' else (4, )
        backbone = __all_masternet__[network_arch](
            structure_info=best_structures[network_id],
            out_indices=out_indices,
            num_classes=1000,
            task=task)
        return backbone, network_arch
    # 将 Timm 原来的 model = create_model(...) 替换为如下
    model, network_arch = get_backbone('best_structure.txt')
    ```

    </details>
    
4. 完成之后，按照 Timm 的使用文档就可以开始自己的训练啦！

5. 在 ImageNet-1k 数据集上 TinyNAS 搜索的网络结构进行训练，在相同计算量和参数量的条件下，网络识别精度有明显提升，如下表所示，top1 accuracy 可以从 77.35% 提升到 79.47%！

    | | top1 | top5 |
    | --- | --- | --- |
    | ResNet-50 | 77.35 | 93.74 |
    | TinyNAS-50 | 79.47 | 94.45 |
'''

def get_classification_interface():
    def run_func(class_num, budget_model_size, budget_flops, budget_layers, ea_num_random_nets, num_network):
        class_num = int(class_num)
        budget_model_size, budget_flops = adjust_data_range(budget_model_size, budget_flops)
        params = locals()
        params["task"] = "classification"
        params["space_arch"] = "MasterNet"
        
        url = client.commit(params)
        if url is None:
            url = 'https://vcs-dockers.oss-cn-hangzhou.aliyuncs.com/TinyNas%2F9d083230-4ae1-11ed-b34b-00163e16c6ef.tar.gz?OSSAccessKeyId=LTAI4GDnHJDwQebpBxVbdEUf&Expires=1982584634&Signature=5zCA0eGwashGfND9TvV78Y8JUHI%3D'
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
        
    description = 'TinyNAS 是一个高性能的神经结构搜索（NAS）框架，用于在GPU和移动设备上自动设计具有高预测精度和高推理速度的深度神经网络。其中，TinyNAS 针对 CNN 网络在分类任务上有着明显的优势——Zen-NAS 方法依靠网络前向推理即可评估网络表达能力，显著降低网络搜索耗时；MAD-NAS 从数学角度评估网络表达能力，无需依赖GPU资源，进一步加速网络搜索性能。'

    class_num = gr.Slider(1, 1000, step =1, value =10, label="Class Num" )
    budget_model_size = gr.Slider(11.69, 50, step= 0.01, value=11.69,label="Max Params (M)")
    budget_flops = gr.Slider(1690, 8e3, step = 10, value=1690, label="Max FLOPs (M)" )
    budget_layers = gr.Slider(16, 200, step=1, value=49, label="Max Layers")
    ea_num_random_nets = gr.Slider(1000, 2000, step=1 , value=1000, label="Iter Num")
    num_network= gr.Slider(1, 5, step=1, value=3, label="How many networks do you want?")
    output_url = gr.TextArea(label="下载链接")
    defaultBtn = """
        <div align="center">
            <div class="gr-button gr-button-lg gr-button-primary" style="background-color: #ddd;">
            下载模型
            </div>
        </div>
    """

    inputs = [class_num, budget_model_size , budget_flops,budget_layers , ea_num_random_nets ,num_network]
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
            [get_classification_interface()],
            ["分类"],
        ) as demo:

        demo.queue(concurrency_count=1)
        demo.launch()
