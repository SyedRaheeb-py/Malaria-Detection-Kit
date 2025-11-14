/*
# -*- coding: utf-8 -*-
Created on Sun Jun 26 19:21:21 2022
@author: SyedRaheeb
*/

$(document).ready(function () {
    $('#result').hide();
    $('.custom-file-input').on('change', function() {
        let fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').addClass("selected").html(fileName);
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $('#btn-predict').prop('disabled', true);

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('#btn-predict').prop('disabled', false);
                $('#result').fadeIn(600);
                $('#diagnosis').text(data);
                console.log('Success!');
            }
        });
    });
});
