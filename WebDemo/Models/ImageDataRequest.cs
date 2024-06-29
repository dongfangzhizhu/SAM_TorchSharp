namespace WebDemo.Models
{
    public class Annotation
    {
        public string Type { get; set; }
        public short X { get; set; }
        public short Y { get; set; }
        public short? X1 { get; set; } // 可以为null，因为不是所有Annotation都有x1和y1
        public short? Y1 { get; set; }
        public short? X2 { get; set; }
        public short? Y2 { get; set; }
    }

    public class ImageDataRequest
    {
        public string Image { get; set; } // 图像的Base64编码
        public List<Annotation> Annotations { get; set; }
    }
}
