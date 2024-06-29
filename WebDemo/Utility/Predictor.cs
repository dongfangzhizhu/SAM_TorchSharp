using SAMTorchSharp;
using TorchSharp;
using WebDemo.Models;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
namespace WebDemo.Utility
{
    public static class Predictor
    {
        private static SamPredictor predictor;
        private static io.SkiaImager skia;
        public static bool IsPedicting { get; private set; }

        static Predictor()
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);
            skia = new torchvision.io.SkiaImager();
            var sam = BuildSam.BuildSAMVitT("./weights/mobile_sam.pt");

            sam.eval();
            predictor = new SamPredictor(sam);
            
        }

        public static byte[] Predict(ImageDataRequest imageDataRequest)
        {
            IsPedicting = true;
            byte[] imageBytes = Convert.FromBase64String(imageDataRequest.Image.Split(',')[1]);
            var image = skia.DecodeImage(imageBytes, io.ImageReadMode.RGB);
            predictor.SetImage(image.unsqueeze(0));
            List<short> boxes=new List<short>();
            List<short> points=new List<short>();
            List<short> pointLabels=new List<short>();
            foreach (var ann in imageDataRequest.Annotations)
            {
                if (ann.Type.Equals("foreground"))
                {
                    points.Add(ann.X);
                    points.Add(ann.Y);
                    pointLabels.Add(1);
                }
                else if(ann.Type.Equals("background"))
                {
                    points.Add(ann.X);
                    points.Add(ann.Y);
                    pointLabels.Add(0);
                }
                else //Type is rectangle
                {
                    boxes.Add(ann.X1.Value);
                    boxes.Add(ann.Y1.Value);
                    boxes.Add(ann.X2.Value);
                    boxes.Add(ann.Y2.Value);
                }
            }

            var input_point = pointLabels.Count>0? torch.tensor(points.ToArray()).reshape(new long[]{-1,2}):null;
            var input_label = pointLabels.Count > 0 ? torch.tensor(pointLabels.ToArray()):null;
            var boxtensor = boxes.Count>0? torch.tensor(boxes):null;
            var (masksNp, iouPredictionsNp, lowResMasksNp) = predictor.Predict(box:boxtensor, pointCoords: input_point, pointLabels: input_label, multimaskOutput: true);
            masksNp = masksNp.squeeze();
            var maskcount = masksNp.shape[0];
            var colors = GenerateDistinctColors((int)maskcount);
            
            for (int i = 0; i < maskcount; i++)
            {
                var imgpixel = ~masksNp[i, TensorIndex.Colon];

                var imgmask = imgpixel.unsqueeze(0).expand_as(image);
                //准备要赋的值,这里假设我们要将masked区域设置为黑色(0)
                var maskcolor = torch.tensor(new short[] { colors[i].R, colors[i].G, colors[i].B }, dtype: ScalarType.Byte).view(3, 1, 1).expand_as(image);
                //使用mask进行赋值
                image = torch.where(imgmask, image, maskcolor);

            }

            IsPedicting = false;
            return skia.EncodeImage(image.to_type(ScalarType.Byte),ImageFormat.Png);
        }
        private static List<(byte R, byte G, byte B)> GenerateDistinctColors(int n)
        {
            List<(byte R, byte G, byte B)> colors = new List<(byte R, byte G, byte B)>();

            for (int i = 0; i < n; i++)
            {
                // 使用黄金比例来生成色相值
                double hue = i * 0.618033988749895 % 1;

                // 固定的饱和度和明度
                double saturation = 0.8; // 80% 饱和度
                double value = 1.0; // 100% 明度

                // 将HSV转换为RGB
                (byte r, byte g, byte b) = HsvToRgb(hue, saturation, value);
                colors.Add((r, g, b));
            }

            return colors;
        }

        private static (byte R, byte G, byte B) HsvToRgb(double h, double s, double v)
        {
            double r, g, b;

            int i = (int)(h * 6);
            double f = h * 6 - i;
            double p = v * (1 - s);
            double q = v * (1 - f * s);
            double t = v * (1 - (1 - f) * s);

            switch (i % 6)
            {
                case 0: r = v; g = t; b = p; break;
                case 1: r = q; g = v; b = p; break;
                case 2: r = p; g = v; b = t; break;
                case 3: r = p; g = q; b = v; break;
                case 4: r = t; g = p; b = v; break;
                default: r = v; g = p; b = q; break;
            }

            return ((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
        }
    }
}
