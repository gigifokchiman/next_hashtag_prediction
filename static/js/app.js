var data = []
var token = ""

jQuery(document).ready(function () {
    var slider = $('#max_words')
    slider.on('change mousemove', function (evt) {
        $('#label_max_words').text('Number of suggestions: ' + slider.val())
    })

    $('#input_text').on('keyup', function (e) {
        if (e.key == ' ') {
            $.ajax({
                url: '/get_end_predictions',
                type: "post",
                contentType: "application/json",
                dataType: "json",
                data: JSON.stringify({
                    "input_text": $('#input_text').val(),
                    "top_k": slider.val(),
                }),
                beforeSend: function () {
                    $('.overlay').show()
                },
                complete: function () {
                    $('.overlay').hide()
                }
            }).done(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
                $('#text_graph_theory').val(jsondata['graph_theory'])
                $('#text_lda').val(jsondata['lda'])
                $('#text_top2vec').val(jsondata['top2vec'])
                $('#text_awd_lstm').val(jsondata['awd_lstm'])
                $('#text_electra').val(jsondata['electra'])
                $('#text_roberta').val(jsondata['roberta'])
            }).fail(function (jsondata, textStatus, jqXHR) {
                console.log(jsondata)
            });
        }
    })

})