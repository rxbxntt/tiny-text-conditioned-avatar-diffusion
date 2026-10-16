import torch

from diffusion.unet import DiffusionUNet

unet = DiffusionUNet(
    time_dim=64,
    context_dim=None,
    num_heads=4,
)

unet.eval()

x = torch.randn(
    2,
    3,
    32,
    32,
)

print("x shape:", x.shape)

time_embedding = torch.randn(
    2,
    64,
)

print("time embedding shape:", time_embedding.shape)

output = unet(
    x=x,
    time_embedding=time_embedding,
    context=None,
)

print("output shape:", output.shape)


assert output.shape == x.shape

assert torch.isfinite(output).all()

print("finite:", torch.isfinite(output).all().item())

loss = output.square().mean()

print("loss:", loss.item())

loss.backward()

has_gradients = False

for name, parameter in unet.named_parameters():
    if parameter.grad is not None:
        has_gradients = True
        print("Gradient found:", name)
        break

assert has_gradients

def main():

    # 1. Create unconditional U-Net
    unet = DiffusionUNet(
        time_dim=64,
        context_dim=None,
        num_heads=4,
    )

    unet.eval()

    # 2. Fake noisy images
    x = torch.randn(
        2,
        3,
        32,
        32,
    )

    # 3. Fake timestep embeddings
    time_embedding = torch.randn(
        2,
        64,
    )

    print("Input:", x.shape)
    print("Time embedding:", time_embedding.shape)

    # 4. Forward pass without text conditioning
    output = unet(
        x=x,
        time_embedding=time_embedding,
        context=None,
    )

    print("Output:", output.shape)

    # 5. Shape check
    assert output.shape == x.shape

    # 6. Numerical stability check
    assert torch.isfinite(output).all()

    # 7. Backward check
    loss = output.square().mean()

    print("Loss:", loss.item())

    loss.backward()

    # 8. Gradient check
    has_gradients = False

    for name, parameter in unet.named_parameters():

        if parameter.grad is not None:

            has_gradients = True

            print("Gradient found:", name)

            break

    assert has_gradients

    print()
    print("UNCONDITIONAL U-NET SANITY CHECK: PASSED")


if __name__ == "__main__":
    main()