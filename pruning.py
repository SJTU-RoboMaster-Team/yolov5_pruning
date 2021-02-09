from models.common import *
from models.yolo import Detect
import argparse
import time


def pruning_conv_output(conv, keep):
    conv.conv.weight.data = conv.conv.weight.data[keep]
    if conv.conv.bias is not None:
        conv.conv.bias.data = conv.conv.bias.data[keep]
    if conv.bn is not None:
        conv.bn.weight.data = conv.bn.weight.data[keep]
        conv.bn.bias.data = conv.bn.bias.data[keep]
        conv.bn.running_var.data = conv.bn.running_var.data[keep]
        conv.bn.running_mean.data = conv.bn.running_mean.data[keep]
    if conv.sp is not None:
        conv.sp.weight.data = conv.sp.weight.data[keep]


def pruning_conv_input(conv, keep):
    conv.conv.weight.data = conv.conv.weight.data[:, keep, ...]


def pruning_conv_transpose_output(conv, keep):
    conv.conv.weight.data = conv.conv.weight.data[:, keep, ...]
    if conv.conv.bias is not None:
        conv.conv.bias.data = conv.conv.bias.data[keep]
    if conv.bn is not None:
        conv.bn.weight.data = conv.bn.weight.data[keep]
        conv.bn.bias.data = conv.bn.bias.data[keep]
        conv.bn.running_var.data = conv.bn.running_var.data[keep]
        conv.bn.running_mean.data = conv.bn.running_mean.data[keep]
    if conv.sp is not None:
        conv.sp.weight.data = conv.sp.weight.data[keep]


def pruning_conv_transpose_input(conv, keep):
    conv.conv.weight.data = conv.conv.weight.data[keep]


def pruning(model, thres):
    modules = model.model._modules
    for k in modules.keys():
        m = modules[k]
        if isinstance(m, Focus):
            keep_output = m.conv.sp.weight.data > thres
            pruning_conv_output(m.conv, keep_output)
        elif isinstance(m, Conv):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            pruning_conv_input(m, keep_input)
            keep_output = m.sp.weight.data > thres
            pruning_conv_output(m, keep_output)
        elif isinstance(m, C3):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            pruning_conv_input(m.cv1, keep_input)
            pruning_conv_input(m.cv2, keep_input)
            keep_output = m.sp.weight.data > thres
            keep_output_bottleneck = keep_output[:m.cv1.conv.weight.shape[0]]
            keep_output_shortcut = keep_output[m.cv1.conv.weight.shape[0]:]
            pruning_conv_output(m.cv1, keep_output_bottleneck)
            pruning_conv_output(m.cv2, keep_output_shortcut)
            for _m in m.m:
                assert isinstance(_m, Bottleneck)
                pruning_conv_input(_m.cv1, keep_output_bottleneck)
                keep_output_inner = _m.cv1.sp.weight.data > thres
                pruning_conv_output(_m.cv1, keep_output_inner)
                keep_input_inner = keep_output_inner
                pruning_conv_input(_m.cv2, keep_input_inner)
                pruning_conv_output(_m.cv2, keep_output_bottleneck)
            m.sp.weight.data = m.sp.weight.data[keep_output]
            pruning_conv_input(m.cv3, keep_output)
            keep_output = m.cv3.sp.weight.data > thres
            pruning_conv_output(m.cv3, keep_output)
        elif isinstance(m, SPP):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            pruning_conv_input(m.cv1, keep_input)
            keep_output = m.cv1.sp.weight.data > thres
            pruning_conv_output(m.cv1, keep_output)
            keep_input = torch.cat([keep_output for i in range(1 + len(m.m))])
            pruning_conv_input(m.cv2, keep_input)
            keep_output = m.cv2.sp.weight.data > thres
            pruning_conv_output(m.cv2, keep_output)
        elif isinstance(m, ConvTranspose):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            pruning_conv_transpose_input(m, keep_input)
            keep_output = m.sp.weight.data > thres
            pruning_conv_transpose_output(m, keep_output)
        elif isinstance(m, Concat):
            keep_input = [modules[str(int(k) - 1) if f == -1 else str(f)].keep_output for f in m.f]
            keep_input = torch.cat(keep_input)
            keep_output = keep_input
        elif isinstance(m, Detect):
            for f, _m in zip(m.f, m.m):
                keep_input = modules[str(int(k) - 1) if f == -1 else str(f)].keep_output
                _m.weight.data = _m.weight.data[:, keep_input, ...]
            keep_output = None
        else:
            assert False, "Unknown layer"
        modules[k].__setattr__("keep_output", keep_output)
    for p in model.parameters():
        p.grad = None


def pruning_present(model, rate):
    modules = model.model._modules
    for k in modules.keys():
        m = modules[k]
        if isinstance(m, Focus):
            thres = torch.sort(m.conv.sp.weight.data).values[int(m.conv.sp.weight.data.shape[0] * rate)]
            keep_output = m.conv.sp.weight.data > thres
            pruning_conv_output(m.conv, keep_output)
        elif isinstance(m, Conv):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            pruning_conv_input(m, keep_input)
            thres = torch.sort(m.sp.weight.data).values[int(m.sp.weight.data.shape[0] * rate)]
            keep_output = m.sp.weight.data > thres
            pruning_conv_output(m, keep_output)
        elif isinstance(m, C3):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            pruning_conv_input(m.cv1, keep_input)
            pruning_conv_input(m.cv2, keep_input)
            thres = torch.sort(m.sp.weight.data).values[int(m.sp.weight.data.shape[0] * rate)]
            keep_output = m.sp.weight.data > thres
            keep_output_bottleneck, keep_output_shortcut = keep_output.view(2, -1)
            pruning_conv_output(m.cv2, keep_output_shortcut)
            pruning_conv_output(m.cv1, keep_output_bottleneck)
            for _m in m.m:
                assert isinstance(_m, Bottleneck)
                pruning_conv_input(_m.cv1, keep_output_bottleneck)
                thres = torch.sort(_m.cv1.sp.weight.data).values[int(_m.cv1.sp.weight.data.shape[0] * rate)]
                keep_output_inner = _m.cv1.sp.weight.data > thres
                pruning_conv_output(_m.cv1, keep_output_inner)
                keep_input_inner = keep_output_inner
                pruning_conv_input(_m.cv2, keep_input_inner)
                pruning_conv_output(_m.cv2, keep_output_bottleneck)
            m.sp.weight.data = m.sp.weight.data[keep_output]
            pruning_conv_input(m.cv3, keep_output)
            thres = torch.sort(m.cv3.sp.weight.data).values[int(m.cv3.sp.weight.data.shape[0] * rate)]
            keep_output = m.cv3.sp.weight.data > thres
            pruning_conv_output(m.cv3, keep_output)
        elif isinstance(m, SPP):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            pruning_conv_input(m.cv1, keep_input)
            thres = torch.sort(m.cv1.sp.weight.data).values[int(m.cv1.sp.weight.data.shape[0] * rate)]
            keep_output = m.cv1.sp.weight.data > thres
            pruning_conv_output(m.cv1, keep_output)
            keep_input = torch.cat([keep_output for i in range(1 + len(m.m))])
            pruning_conv_input(m.cv2, keep_input)
            thres = torch.sort(m.cv2.sp.weight.data).values[int(m.cv2.sp.weight.data.shape[0] * rate)]
            keep_output = m.cv2.sp.weight.data > thres
            pruning_conv_output(m.cv2, keep_output)
        elif isinstance(m, nn.ConvTranspose2d):
            keep_input = modules[str(int(k) - 1) if m.f == -1 else str(m.f)].keep_output
            m.weight.data = m.weight.data[keep_input]
            keep_output = torch.ones_like(m.bias.data).bool().to(m.bias.device)
        elif isinstance(m, Concat):
            keep_input = [modules[str(int(k) - 1) if f == -1 else str(f)].keep_output for f in m.f]
            keep_input = torch.cat(keep_input)
            keep_output = keep_input
        elif isinstance(m, Detect):
            for f, _m in zip(m.f, m.m):
                keep_input = modules[str(int(k) - 1) if f == -1 else str(f)].keep_output
                _m.weight.data = _m.weight.data[:, keep_input, ...]
            keep_output = None
        else:
            assert False, "Unknown layer"
        modules[k].__setattr__("keep_output", keep_output)
    for p in model.parameters():
        p.grad = None


def summary(model):
    for m in model.model:
        if isinstance(m, Focus):
            co, ci = m.conv.conv.weight.shape[:2]
            print(f"Focus  input:{ci:4} output:{co:4}")
        elif isinstance(m, Conv):
            co, ci = m.conv.weight.shape[:2]
            print(f"Conv   input:{ci:4} output:{co:4}")
        elif isinstance(m, C3):
            co_b, ci = m.cv1.conv.weight.shape[:2]
            co_s = m.cv2.conv.weight.shape[0]
            co = m.cv3.conv.weight.shape[0]
            print(f"C3     input:{ci:4} output:{co:4} bottleneck:{co_b:4} shortcut:{co_s:4}")
        elif isinstance(m, SPP):
            co_p, ci = m.cv1.conv.weight.shape[:2]
            co = m.cv2.conv.weight.shape[0]
            print(f"SPP    input:{ci:4} output:{co:4} inner:{co_p:4}")
        elif isinstance(m, nn.ConvTranspose2d):
            ci, co = m.weight.shape[:2]
            print(f"ConvTr input:{ci:4} output:{co:4}")
        elif isinstance(m, Concat):
            # ci = co = torch.sum(m.keep_output).item()
            print(f"Concat")
        elif isinstance(m, Detect):
            ci_0 = m.m[0].weight.shape[1]
            ci_1 = m.m[1].weight.shape[1]
            ci_2 = m.m[2].weight.shape[1]
            print(f"Detect input_0:{ci_0:4} input_1:{ci_1:4} input_2:{ci_2:4}")


def cnt_time(model, *args):
    t1 = time.time()
    with torch.no_grad():
        for i in range(100):
            _ = model(*args)
    t2 = time.time()
    return (t2 - t1) / 100.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="weights to be pruning")
    parser.add_argument("--threshold", type=float, default=1e-3, help="pruning threshold")
    opt = parser.parse_args()
    x = torch.randn([1, 3, 384, 640]).cpu()
    pt_file = torch.load(opt.weights)
    model = pt_file["model"].float().cpu()
    print(f"before pruning: {cnt_time(model, x)}")
    summary(model)
    pruning(model, opt.threshold)
    print(f"after pruning: {cnt_time(model, x)}")
    summary(model)
    pt_file["model"] = model
    torch.save(pt_file, opt.weights.replace(".pt", "_pruning.pt"))
