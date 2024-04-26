function makePrediction() {
    var textInput = $('#textInput').val();  // Get the value from the textarea
    if (textInput) {
        $.ajax({
            url: '/predict',  // Endpoint where the Flask app handles predictions
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ text: textInput }),  // Convert text input into JSON
            // read response and output to the user
            success: function(response) {
                $('#predictionResult').html('Prediction: ' + response.prediction);
            },
            error: function(error) {
                console.log('Error:', error);
                $('#predictionResult').html('Failed to get prediction.');
            }
        });
    } else {
        $('#predictionResult').html('Please enter some text.');
    }
}
