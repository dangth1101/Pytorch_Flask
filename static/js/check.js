function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function(e) {
            var col = $(input).closest('.col');
            col.find('.image-upload-wrap').hide();
            col.find('.file-upload-image').attr('src', e.target.result);
            col.find('.file-upload-content').show();
        };

        reader.readAsDataURL(input.files[0]);
    } else {
        removeUpload(input);
    }
}

function removeUpload(input) {
    var col = $(input).closest('.col');
    col.find('.file-upload-input').replaceWith($(input).clone()); 
    col.find('.file-upload-content').hide(); 
    col.find('.image-upload-wrap').show(); 
    $(input).val(''); 
    document.querySelector('.prediction-display').textContent = '';
}

function shouldShowCheckIcon(key, value) {
    switch (key) {
        case 'ssim':
            return value > 0.9999999 && value <= 1;
        case 'psnr':
            return value >= 40;
        case 'l0':
        case 'l1':
        case 'l2':
            return value < 1;
        default:
            return false;
    }
}


$(document).ready(function() {
    $('.check-btn').click(function(e) {
        e.preventDefault();

        let formData = new FormData();
        formData.append('file1', $('input.file-upload-input-left')[0].files[0]);
        formData.append('file2', $('input.file-upload-input-right')[0].files[0]);

        $("#results-table tbody").empty();
        // Send the AJAX request
        $.ajax({
            url: '/check_result', // The endpoint to handle image processing
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                allSuccessful = true;
                for (let key in data) {
                    let value = parseFloat(data[key]).toFixed(21);
                    let success = shouldShowCheckIcon(key, value);
                    
                    if (!success) {
                        allSuccessful = false;
                    }
        
                    let iconName = success ? 'fa-check text-success' : 'fa-exclamation-triangle text-danger';
        
                    // Create table row
                    let row = $('<tr>')
                        .append($('<td>').text(key))
                        .append($('<td>').text(value))
                        .append($('<td>').html($('<i>', {class: 'fas ' + iconName})));
        
                    // Append the row to the table body
                    $("#results-table tbody").append(row);
                }
        
                // Append the overall status as a row
                let overallStatusText = allSuccessful ? 'Success' : 'Fail';
                let overallStatusColorClass = allSuccessful ? 'text-success' : 'text-danger';
        
                let overallStatusRow = $('<tr>')
                    .append($('<td>', {colspan: 2}).text('Overall Status'))
                    .append($('<td>').html($('<span>', {class: overallStatusColorClass}).text(overallStatusText)));
        
                $("#results-table tbody").append(overallStatusRow);
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.log(jqXHR.responseText); 
                console.log(textStatus); 
                console.log(errorThrown); 
            }
        });
    });
});


