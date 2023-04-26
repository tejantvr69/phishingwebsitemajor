$(document).ready(function() {
	$('#form').submit(function(e) {
		e.preventDefault();
		var url = $('#url').val();
		$.ajax({
			url: 'http://your-api-endpoint.com/check-url',
			method: 'POST',
			data: {url: url},
			success: function(response) {
				// Display the results
				if (response.is_phishing) {
					$('#result').html('<p>This website is a phishing website.</p>');
				} else {
					$('#result').html('<p>This website is safe.</p>');
				}
			},
			error: function() {
				// Handle errors
				alert('Error: Failed to check the URL.');
			}
		});
	});
});
