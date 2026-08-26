namespace WebDemo.Models
{
    public class Annotation
    {
        public string Type { get; set; } = string.Empty;
        public int X { get; set; }
        public int Y { get; set; }
        public int? X1 { get; set; } // 可以为null，因为不是所有Annotation都有x1和y1
        public int? Y1 { get; set; }
        public int? X2 { get; set; }
        public int? Y2 { get; set; }
    }

    public class ImageDataRequest
    {
        public string Model { get; set; } = "sam1";
        public string Image { get; set; } = string.Empty; // 图像的Base64编码
        public string? Caption { get; set; }
        public List<Annotation> Annotations { get; set; } = [];
    }
}
