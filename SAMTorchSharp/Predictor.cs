using SAMTorchSharp.Modeling;
using SAMTorchSharp.Utils;
using static TorchSharp.torch;

namespace SAMTorchSharp
{
    public class SamPredictor
    {
        private Sam model;
        private ResizeLongestSide transform;
        private Tensor features;
        private bool isImageSet;
        private long[] originalSize;
        private long[] inputSize;

        public SamPredictor(Sam samModel)
        {
            model = samModel;
            transform = new ResizeLongestSide(samModel.image_encoder.imgSize);
            reset_image();
        }

        public void SetImage(Tensor image)
        {
            Tensor inputImage = transform.ApplyImage(image);
            SetTorchImage(inputImage, image.shape[2], image.shape[3]);
        }

        private void SetTorchImage(Tensor transformedImage, long originalHeight, long originalWidth)
        {
            // Check the shape of the image tensor
            var transformedImgShape = transformedImage.shape;
            if (transformedImgShape.Length != 4 || transformedImgShape[1] != 3 || Math.Max(transformedImgShape[2], transformedImgShape[3]) != model.image_encoder.imgSize)
            {
                throw new ArgumentException("set_torch_image input must be BCHW with long side matching the model's img_size.");
            }

            reset_image();

            originalSize = new long[]{originalHeight, originalWidth};
            inputSize = new long[] { transformedImgShape[2], transformedImgShape[3]};

            Tensor inputImage = model.preprocess(transformedImage);
            features = model.image_encoder.forward(inputImage);
            isImageSet = true;
        }

        public (Tensor, Tensor, Tensor) Predict(Tensor pointCoords = null, Tensor pointLabels = null, Tensor box = null, Tensor maskInput = null, bool multimaskOutput = true, bool returnLogits = false)
        {
            if (!isImageSet)
            {
                throw new InvalidOperationException("An image must be set with .set_image(...) before mask prediction.");
            }

            // Transform input prompts to Tensors if provided
            Tensor coordsTorch =null, labelsTorch=null, boxTorch=null, maskInputTorch = null;
            if (pointCoords is not null)
            {
                if (pointLabels is null)
                {
                    throw new InvalidOperationException("pointLabels must be supplied if pointCoords is supplied.");
                }
                // Apply transformation to point coordinates and labels
                // This is a placeholder for the actual transformation logic
                pointCoords = transform.ApplyCoords(pointCoords, (originalSize[0], originalSize[1]));
                coordsTorch = pointCoords.to(Device).unsqueeze(0);
                labelsTorch = pointLabels.to(Device).unsqueeze(0);
            }
            if (box is not null)
            {
                // Apply transformation to boxes
                // This is a placeholder for the actual transformation logic
                box = transform.ApplyBoxes(box, (originalSize[0], originalSize[1]));
                boxTorch = box.to(Device).unsqueeze(0);
            }
            if (maskInput is not null)
            {
                maskInputTorch = maskInput.to(Device).unsqueeze(0);
            }

            // Predict masks using the model
            var (masks, iouPredictions, lowResMasks) = predict_torch(coordsTorch, labelsTorch, boxTorch, maskInputTorch, multimaskOutput, returnLogits);


            return (masks, iouPredictions, lowResMasks);
        }

        private (Tensor, Tensor, Tensor) predict_torch(Tensor pointCoords, Tensor pointLabels, Tensor boxes, Tensor maskInput, bool multimaskOutput, bool returnLogits)
        {
            if (!isImageSet)
            {
                throw new InvalidOperationException("An image must be set with .set_image(...) before mask prediction.");
            }

            // Embed prompts
            var (sparseEmbeddings, denseEmbeddings) = model.prompt_encoder.forward(
                points: pointCoords is null?null : Tuple.Create(pointCoords, pointLabels),
                boxes: boxes,
                masks: maskInput
            );
            var dense_pe = model.prompt_encoder.get_dense_pe();
            // Predict masks
            var (lowResMasks, iouPredictions) = model.mask_decoder.forward(
                imageEmbeddings: features,
                imagePe: dense_pe,
                sparsePromptEmbeddings: sparseEmbeddings,
                densePromptEmbeddings: denseEmbeddings,
                multimaskOutput: multimaskOutput
            );
            

            // Upscale the masks to the original image resolution
            Tensor masks = model.postprocess_masks(lowResMasks, inputSize, originalSize);

            if (!returnLogits)
            {
                masks = masks > model.mask_threshold;
            }

            return (masks, iouPredictions, lowResMasks);
        }

        public Tensor GetImageEmbedding()
        {
            if (!isImageSet)
            {
                throw new InvalidOperationException("An image must be set with .set_image(...) to generate an embedding.");
            }
            if (features is null)
            {
                throw new InvalidOperationException("Features must exist if an image has been set.");
            }
            return features;
        }

        public Device Device => model.device();

        private void reset_image()
        {
            isImageSet = false;
            features = null;
            originalSize = null;
            inputSize = null;
        }
    }
}
