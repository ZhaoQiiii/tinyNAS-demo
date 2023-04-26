import gradio as gr
from common import *

markdown = r'''
### 使用方法

1. 资源设定网络

    用户根据自己的场景需要，将诸如网络参数量，计算量等资源限制告知系统，TinyNAS 就会搜索满足用户要求的网络。在本demo 中，用户只需要通过拖动条进行设置即可，具体参数的定义解释如下表所示：

    | 序号 | 约束| 定义解释 | 
    | --- | --- | --- | 
    | 1	| Max Params（M) | 搜索出来的各种网络结构中，其参数量最大不能超过多少（参考值：9.64-62.63M） | 
    | 2	| Max FLOPs（M)	| 搜索出来的各种网络结构中，其浮点运算次数最大不能超过多少（参考值：7.75-138.99G） | 
    | 3	| Max Layers | 搜索出来的各种网络结构中，其网络层数（也表示网络深度）最大不能超过多少（参考值：16-200） | 
    | 4	| Iter Num | 搜索网络结构时搜索的迭代次数，可以认为是搜索时长的另一种体现（参考值：1000-2000） | 

2. 执行搜索

    在资源设定完成后，用户只需要点击提交，系统就会执行搜索，完毕后在右侧就会返回搜索到的网络结构及其使用示例，用户下载即可使用。

3. 下载搜索结果 tar 包之后，用户执行以下命令，即可对搜索得到的网络结构进行一次前向推理

    <details>

    <summary>查看代码</summary>
    
    ```bash
    tar xzvf xxx.tar.gz
    cd 解压目录
    python3 demo.py
    ```

    </details>

    用户可以基于demo.py中的示例代码方便地将 NAS 网络集成到训练流程中。

### 一个例子：在GFocalV2中使用搜索结果
1. 首先按照GFocalV2的<a href="https://github.com/implus/GFocalV2">官方代码库</a>对其进行安装。

2. 将搜索结果解压目录中的modules目录和masternet.py文件拷贝到GFocalV2目录下的mmdet/backbones路径，并将masternet.py中的16、17行从modules导入包的语句改为相对引用

    <details>
    <summary>查看代码</summary>
    
    ```bash
    cp -r <解压目录>/modules <GFocalV2目录>/mmdet/backbones/
    cp -r <解压目录>/masternet.py <GFocalV2目录>/mmdet/backbones/
    # masternet.py中的16、17行修改为：
    from .modules import __all_blocks__, network_weight_stupid_init
    from .modules.qconv import QLinear
    ```
    
    </details>

3. 在GFocalV2目录下的configs路径中创建一个用于存放maedet配置文件的目录gfocal_maedet，并将搜索结果解压目录中搜索到的结构文本文件best_structure.json拷贝到配置目录中

    <details>
    <summary>查看代码</summary>

    ```bash
    mkdir -p <GFocalV2目录>/configs/gfocal_maedet
    cp <解压目录>/best_structure.json <GFocalV2目录>/configs/gfocal_maedet
    ```
    
    </details>

4. 将demo.py复制到 GFocalV2 中存放 backbone 的目录mmdet/backbones并重命名为tinynas.py，同时将其中的get_backbone函数包装为一个能被 GFocalV2 使用的 TinyNAS 类

    <details>
    <summary>查看代码</summary>

    ```bash
    cp <解压目录>/demo.py <GFocalV2目录>/mmdet/backbones/tinynas.py
    ```

    </details>
    
    包装后的tinynas.py代码
    
    <details>
    <summary>查看代码</summary>
    
    ```python
    # Copyright (c) Alibaba, Inc. and its affiliates. 
    import ast
    import os
    import torch
    #需要在 masternet 前添加 . 表示从当前目录导入
    from .masternet import MasterNet
    ################################
    #                              #
    #        定义 TinyNAS 类        #
    #                              #
    ################################
    import torch.nn as nn
    from ..builder import BACKBONES
    from mmcv.cnn import ConvModule, constant_init, kaiming_init
    from torch.nn.modules.batchnorm import _BatchNorm
    @BACKBONES.register_module
    class TinyNAS(nn.Module):
        def __init__(self, net_str=None):
            super(TinyNAS, self).__init__()
            self.body, _ = get_backbone(
                net_str,
                load_weight=False,
                task='detection')
        def init_weights(self, pretrained=None):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        def forward(self, x):
            """Forward function."""
            return self.body(x)
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
        # If task type is classification, param num_classes is required
        out_indices = (1, 2, 3, 4) if task == 'detection' else (4, )
        backbone = MasterNet(
                structure_info=best_structures[network_id],
                out_indices=out_indices,
                num_classes=1000,
                task=task)
        return backbone, network_arch
    ################################
    #                              #
    #         测试前向推理           #
    #                              #
    ################################
    if __name__ == '__main__':
        # make input
        x = torch.randn(1, 3, 224, 224)
        # instantiation
        backbone, network_arch = get_backbone('best_structure.json', task='detection')
        print(backbone)
        # forward
        input_data = [x]
        pred = backbone(*input_data)
        #print output
        for o in pred:
            print(o.size())
    ```

    </details>

5. 将以下代码片段添加到 <GFocalV2目录>/mmdet/backbones/\_\_init__.py，以此将新添加的 TinyNAS 类添加到可用 backbone 列表中

    <details>

    <summary>查看代码</summary>

    ```python
    from .tinynas import TinyNAS # add this
    __all__ = [
        'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net','HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet', 
        "TinyNAS" # add this
    ]
    ```

    </details>

6. 将以下代码片段添加到训练代码 <GFocalV2目录>/tools/train.py 中的模型构建之后，即153行

    <details>

    <summary>查看代码</summary>

   ```python
    if cfg.model.backbone.type in ["TinyNAS"] and cfg.use_syncBN_torch:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info(f'Model:\n{model}')
    ```

    </details>

7. 最后，按照 GFocalV2 的使用文档，将 GFocalV2 模型配置文件中的 backbone 替换为 TinyNAS，就可以开始自己的训练啦！

8. 在 COCO 数据集上使用 GFVL2 框架训练，TinyNAS 检测模型相较于 ResNet-50 在近似的 FLOPs 下有近2.0%的提升，而在相同精度的情况下有1.54倍的推理速度提升

    | Backbone | Param (M) | FLOPs (G) | box APval | box APS | box APM | box APL |
    | --- | --- | --- | --- | --- | --- | --- |
    | ResNet-50 | 23.5 | 83.6 | 44.7 | 29.1 | 48.1 | 56.6 |
    | TinyNAS-DET-S | 21.2 | 48.7 | 45.1 | 27.9 | 49.1 | 58.0 |
    | TinyNAS-DET-M | 25.8 | 89.9 | 46.9 | 30.1 | 50.9 | 59.9 |
'''

def get_detection_interface():
    def run_func( budget_model_size, budget_flops, budget_layers, ea_num_random_nets, num_network):
        budget_model_size, budget_flops = adjust_data_range(budget_model_size, budget_flops)
        params = locals()
        params["task"] = "detection"
        params["space_arch"] = "MasterNet"
        
        url = client.commit(params)
        if url is None:
            url = 'https://vcs-dockers.oss-cn-hangzhou.aliyuncs.com/TinyNas%2Fe822e724-4ae1-11ed-858f-00163e16c6ef.tar.gz?OSSAccessKeyId=LTAI4GDnHJDwQebpBxVbdEUf&Expires=1982584673&Signature=OjvNRGdmbAUHoB9kDVnImVJXB1I%3D'
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
        
    description = 'TinyNAS 是一个高性能的神经结构搜索（NAS）框架，用于在GPU和移动设备上自动设计具有高预测精度和高推理速度的深度神经网络。其中，面向检测任务的TinyNAS服务采用了阿里巴巴达摩院发表在ICML22上的MaE-DET方法作为网络结构评估算法，该方法只需要10-20分钟，即可根据用户的需求设计出相应的目标检测网络。搜索出的结构可以很好的适配 YOLO，GFOCALV2，RetinaNet，FCOS 等主流目标检测框架，并带来明显的精度提升。目前该演示只支持CNN搜索空间'

    budget_model_size = gr.Slider(9.65, 62.63, step= 0.01, value=9.65,label="Max Params (M)")
    budget_flops = gr.Slider(7750, 138.99e3, step = 10, value=7750, label="Max FLOPs (M)" )
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

    inputs = [budget_model_size , budget_flops,budget_layers , ea_num_random_nets ,num_network ]
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
            [get_detection_interface()],
            ["Search on Detection"],
        ) as demo:

        demo.queue(concurrency_count=1)
        demo.launch()
