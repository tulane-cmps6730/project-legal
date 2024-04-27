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
                let results = []
                console.log(response.prediction)
                console.log(response.text)
                for (let i = 0; i < response.prediction.length; i++) {
                    if (response.prediction[i] >= 0.5) {
                        console.log(response.text[i])
                        results.push(response.text[i])   
                    }
                }
                console.log(results)
                if (results.length == 0) {
                    $('#predictionResult').html('We did not identify any sentences as potentially unfair or exploitative.');
                }
                else {
                    $('#predictionResult').html('We identified the following sentence(s) as potentially unfair or exploitative:\n');
                    $.each(results, function(index, value) {
                        $('#positiveSentences').append('<li>' + value + '</li>');
                    });
                }
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
