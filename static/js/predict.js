function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
      $('.image-upload-wrap').hide();

      $('.file-upload-image').attr('src', e.target.result);
      $('.file-upload-content').show();

      $('.image-title').html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);

  } else {
    removeUpload();
  }
}

function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
  $('.file-upload-input').val(''); 
  document.querySelector('.prediction-display').textContent = '';
}

$(document).ready(function() {
  $('.predict-image').click(function(e) {
      e.preventDefault();
      
      var formData = new FormData();
      formData.append('file', $('.file-upload-input')[0].files[0]);

      var selectedModel = $('#model').val();
      formData.append('selectedModel', selectedModel); 
      
      $.ajax({
          type: 'POST',
          url: '/predict_result',
          data: formData,
          processData: false,
          contentType: false,
          success: function(data) {
              // Convert the data dictionary into an array of [class, probability] pairs
              var items = Object.keys(data).map(function(key) {
                  return [key, data[key]];
              });

              // Sort the array by the probabilities in descending order
              items.sort(function(first, second) {
                  return second[1] - first[1];
              });

              var results = '<h2>Class Probabilities:</h2><ul>';
              for (var i = 0; i < items.length; i++) {
                  results += `<li>${items[i][0]}: ${items[i][1].toFixed(2)}%</li>`;
              }
              results += '</ul>';
              $('.prediction-display').html(results);
          }
      });
  });
});

