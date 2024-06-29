using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using WebDemo.Models;
using WebDemo.Utility;

namespace WebDemo.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }
        [HttpPost]
        public IActionResult Predict([FromBody] ImageDataRequest imageDataRequest)
        {
            var fileBytes = Predictor.Predict(imageDataRequest);
            return new FileStreamResult(new MemoryStream(fileBytes), "application/octet-stream");
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
