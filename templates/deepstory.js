$(document).ready(function () {
    function refresh(){
        $("#status").load("{{ url_for('status') }}");
        $("#sentences").load("{{ url_for('sentences') }}");
        $("#gpt2").load("{{ url_for('gpt2') }}", function() {
            $('textarea').each(function() {
                if($(this).closest(".tab-pane").hasClass('active')){
                    this.style.height = 'auto';
                    this.style.height = (this.scrollHeight) + 'px';
                }
            }).on('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
        });
    }
    refresh();
    // so that the scrollHeight of GPT2 textarea will only be calculated when it is active, otherwise 0
    $("a.nav-link").click(function(){
        $('.tab-pane').each(function(){
            if($(this).hasClass('active')) {
                $(this).find('textarea').each(function() {
                    this.style.height = 'auto';
                    this.style.height = (this.scrollHeight) + 'px';
                });
            }
        });
    });
    $("#animate").find("select").each(function(){
        let img = $('<img />', {src: "image/" + $(this).val()});
        img.insertAfter($(this));
    }).on('change', function(){
        $(this).parent().find("img").attr('src', "image/" + $(this).val());
    }).submit(function (e) {
        e.preventDefault();
        $.ajax({
            type: "GET",
            url: this.action,
            data: $(this).serialize(),
            success: function (message) {
                alert(message);
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
            }
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
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#loadGPT2").submit(function (e) {
        e.preventDefault();
        $.ajax({
            type: "GET",
            url: this.action,
            data: $(this).serialize(),
            success: function (message) {
                alert(message);
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#generate").submit(function (e) {
        e.preventDefault();
        $.ajax({
            type: "POST",
            url: this.action,
            data: $(this).serialize(),
            success: function (message) {
                alert(message);
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
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
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#synthesize").click(function () {
        $.ajax({
            url: "{{ url_for('synthesize') }}",
            success: function (message) {
                alert(message);
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#createAudio").click(function () {
        $.ajax({
            url: "{{ url_for('combine') }}",
            success: function (message) {
                alert(message);
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#createVideo").click(function () {
        $.ajax({
            url: "{{ url_for('create_base') }}",
            success: function (message) {
                alert(message);
                refresh();
            },
            error: function (response) {
                alert(response.responseText);
            }
        });
    });
    $("#view").click(function () {
        if (!$(this).parent().has('video').length) {
            let video = $('<video />', {
                src: "{{ url_for('video') }}",
                type: 'video/mp4',
                controls: true
            });
            video.insertAfter($(this));
        } else {
            $(this).parent().find("video").attr('src', "{{ url_for('video') }}");
        }
    });
});