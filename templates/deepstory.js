$(document).ready(function () {
    function refresh_status() {
        $("#status").load("{{ url_for('status') }}", function () {
            $("#clearCache").click(function () {
                $.ajax({
                    url: "{{ url_for('clear') }}",
                    success: function (message) {
                        alert(message);
                        refresh_status();
                        refresh_animate();
                        refresh_video();
                    },
                    error: function (response) {
                        alert(response.responseText);
                    }
                });
            });
        });
    }
    function refresh_sent() {
        $("#sentences").load("{{ url_for('sentences') }}");
    }
    function refresh_animate() {
        $("#tab-3").load("{{ url_for('animate') }}", function () {
            $("#animate").find("select").each(function () {
                let img = $('<img />', {src: "image/" + $(this).val()});
                img.insertAfter($(this));
            }).on('change', function () {
                $(this).parent().find("img").attr('src', "image/" + $(this).val());
            });
            $("#animate").submit(function (e) {
                e.preventDefault();
                let button = $(this).find('button')
                let tempText = button.text();
                button.text("Animating...");
                button.prop('disabled', true);
                $.ajax({
                    type: "POST",
                    url: this.action,
                    data: $(this).serialize(),
                    success: function (message) {
                        alert(message);
                        button.text(tempText);
                        button.prop('disabled', false);
                        refresh_status();
                        refresh_video();
                    },
                    error: function (response) {
                        alert(response.responseText);
                        button.text(tempText);
                        button.prop('disabled', false);
                    }
                });
            });
        });
    }
    function refresh_video() {
        $("#tab-4").load("{{ url_for('video') }}", function () {
            $("#view").click(function () {
                if (!$(this).parent().has('video').length) {
                    let video = $('<video />', {
                        src: "{{ url_for('video_viewer') }}",
                        type: 'video/mp4',
                        controls: true
                    });
                    video.insertAfter($(this));
                } else {
                    $(this).parent().find("video").attr('src', "{{ url_for('video_viewer') }}");
                }
            });
        });
    }
    function refresh_map() {
        $("#mapCard").load("{{ url_for('map_page') }}", function () {
            $("#mapSpeaker").submit(function (e) {
                e.preventDefault();
                $.ajax({
                    type: "POST",
                    url: this.action,
                    data: $(this).serialize(),
                    success: function (message) {
                        alert(message);
                        refresh_map();
                        refresh_sent();
                    },
                    error: function (response) {
                        alert(response.responseText);
                    }
                });
            });
        });
    }
    function refresh_gpt2() {
        $("#gpt2").load("{{ url_for('gpt2') }}", function () {
            $('.tab-pane').each(function () {
                $(this).trigger('fixText')
            });
        });
    }
    function refresh_add_sent() {
        $("#sentenceCard").load("{{ url_for('gen_sents') }}", function() {
            $("button[class*='add']").click(function (e) {
                e.preventDefault();
                $.ajax({
                    type: "GET",
                    url: "{{ url_for('add_sent') }}",
                    data: this.name + '=' + this.value,
                    success: function (message) {
                        alert(message);
                        refresh_gpt2();
                        refresh_add_sent();
                    },
                    error: function (response) {
                        alert(response.responseText);
                    }
                });
            });
        });
    }
    $('textarea').each(function() {
        if($(this).closest(".tab-pane").hasClass('active')){
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        }
    });
    refresh_status();
    refresh_sent();
    refresh_animate();
    refresh_video();
    refresh_map();
    refresh_gpt2();
    refresh_add_sent();

    $("#gpt2Card").hide();
    $("#sentenceCard").hide();
    // so that the scrollHeight of GPT2 textarea will only be calculated when it is active, otherwise 0
    $('.tab-pane').on('fixText', function() {
        if($(this).hasClass('active')){
            $(this).find('textarea').each(function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            }).on('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
        }
    })
    $('.tab-pane').each(function() {
        $(this).trigger('fixText')
    });
    $('.nav-tabs a').on('shown.bs.tab', function(){
        if (this.text === 'GPT2') {
            $("#gpt2Card").show();
            $("#sentenceCard").show();
            $("#status").hide();
            $("#sentenceArea").hide();
        } else {
            $("#gpt2Card").hide();
            $("#sentenceCard").hide();
            $("#status").show();
            $("#sentenceArea").show();
        }
        $('.tab-pane').each(function() {
            $(this).trigger('fixText')
        });
    });
    $("#loadSent").submit(function (e) {
        e.preventDefault();
        $.ajax({
            type: "POST",
            url: this.action,
            data: $(this).serialize(),
            success: function (message) {
                alert(message);
                refresh_sent();
                refresh_map();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#loadText").submit(function (e) {
        e.preventDefault();
        $.ajax({
            type: "GET",
            url: this.action,
            data: $(this).serialize(),
            success: function (message) {
                refresh_gpt2();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#loadGPT2").submit(function (e) {
        e.preventDefault();
        let button = $(this).find('button')
        let tempText = button.text();
        button.text("Loading...");
        button.prop('disabled', true);
        $.ajax({
            type: "GET",
            url: this.action,
            data: $(this).serialize(),
            success: function (message) {
                alert(message);
                button.text(tempText);
                button.prop('disabled', false);
                refresh_gpt2();
                refresh_add_sent();
            },
            error: function (response) {
                alert(response.responseText);
                button.text(tempText);
                button.prop('disabled', false);
            }
        });
    });
    $("#generateText").submit(function (e) {
        e.preventDefault();
        let button = $(this).find('button')
        let tempText = button.text();
        button.text("Generating...");
        button.prop('disabled', true);
        $.ajax({
            type: "POST",
            url: this.action,
            data: $.merge($(this).serializeArray(), $("#gpt2Text").serializeArray()),
            success: function (message) {
                alert(message);
                button.text(tempText);
                button.prop('disabled', false);
                refresh_gpt2();
                refresh_add_sent();
            },
            error: function (response) {
                alert(response.responseText);
                button.text(tempText);
                button.prop('disabled', false);
            }
        });
    });
    $("#generateSentence").submit(function (e) {
        e.preventDefault();
        let button = $(this).find('button')
        let tempText = button.text();
        button.text("Predicting...");
        button.prop('disabled', true);
        $.ajax({
            type: "POST",
            url: this.action,
            data: $.merge($.merge($(this).serializeArray(), $("#generateText").serializeArray()), $("#gpt2Text").serializeArray()),
            success: function (message) {
                alert(message);
                button.text(tempText);
                button.prop('disabled', false);
                refresh_add_sent();
            },
            error: function (response) {
                alert(response.responseText);
                button.text(tempText);
                button.prop('disabled', false);
            }
        });
    });
    $("#modify").click(function () {
        const data = $('tbody tr').map(function() {
            let obj = [];
            $(this).find('select').each(function() {
                obj.push($(this).val());
            });
            return obj;
        }).get();
        $.ajax({
            type: "POST",
            url: "{{ url_for('modify') }}",
            contentType: "application/json",
            data: JSON.stringify(data),
            success: function (message) {
                alert(message);
                refresh_sent();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#synthesize").click(function () {
        let button = $(this);
        let tempText = button.text();
        button.text("Synthesizing...");
        button.prop('disabled', true);
        $.ajax({
            url: "{{ url_for('synthesize') }}",
            success: function (message) {
                alert(message);
                button.text(tempText);
                button.prop('disabled', false);
                refresh_status();
                refresh_sent();
            },
            error: function (response) {
                alert(response.responseText);
                button.text(tempText);
                button.prop('disabled', false);
            }
        });
    });
    $("#createAudio").click(function () {
        let button = $(this);
        let tempText = button.text();
        button.text("Creating...");
        button.prop('disabled', true);
        $.ajax({
            url: "{{ url_for('combine') }}",
            success: function (message) {
                alert(message);
                button.text(tempText);
                button.prop('disabled', false);
                refresh_status();
            },
            error: function (response) {
                alert(response.responseText);
                button.text(tempText);
                button.prop('disabled', false);
            }
        });
    });
    $("#createVideo").click(function () {
        let button = $(this);
        let tempText = button.text();
        button.text("Creating...");
        button.prop('disabled', true);
        $.ajax({
            url: "{{ url_for('create_base') }}",
            success: function (message) {
                alert(message);
                button.text(tempText);
                button.prop('disabled', false);
                refresh_status();
                refresh_animate();
            },
            error: function (response) {
                alert(response.responseText);
                button.text(tempText);
                button.prop('disabled', false);
            }
        });
    });
});