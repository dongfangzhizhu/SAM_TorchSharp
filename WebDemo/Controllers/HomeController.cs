using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using WebDemo.Models;
using WebDemo.Utility;

namespace WebDemo.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly ModelInferenceService _inference;

        public HomeController(ILogger<HomeController> logger, ModelInferenceService inference)
        {
            _logger = logger;
            _inference = inference;
        }

        public IActionResult Index()
        {
            return View(_inference.GetAvailability());
        }
        [HttpPost]
        [RequestSizeLimit(25 * 1024 * 1024)]
        public async Task<IActionResult> Predict([FromBody] ImageDataRequest imageDataRequest, CancellationToken cancellationToken)
        {
            try
            {
                return Ok(await _inference.PredictAsync(imageDataRequest, cancellationToken));
            }
            catch (Exception exception) when (exception is ArgumentException or FileNotFoundException or InvalidDataException)
            {
                _logger.LogWarning(exception, "Model inference request was rejected.");
                return BadRequest(new { error = exception.Message });
            }
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
