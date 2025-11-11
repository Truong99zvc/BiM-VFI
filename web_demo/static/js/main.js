$(document).ready(function() {
    // Xem trước ảnh
    function previewImage(input, previewId) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            
            reader.onload = function(e) {
                const img = $('<img>').attr('src', e.target.result);
                $(`#${previewId}`).empty().append(img);
            }
            
            reader.readAsDataURL(input.files[0]);
        }
    }
    
    // Xử lý nút xem trước
    $('#preview1-btn').click(function() {
        $('#image1').click();
    });
    
    $('#preview2-btn').click(function() {
        $('#image2').click();
    });
    
    // Xử lý khi chọn file ảnh
    $('#image1').change(function() {
        previewImage(this, 'image1-preview');
    });
    
    $('#image2').change(function() {
        previewImage(this, 'image2-preview');
    });
    
    // Xử lý form submit
    $('#upload-form').submit(function(e) {
        e.preventDefault();
        
        // Kiểm tra xem đã chọn cả hai ảnh chưa
        if (!$('#image1')[0].files[0] || !$('#image2')[0].files[0]) {
            showError('Vui lòng chọn cả hai ảnh đầu vào');
            return;
        }
        
        // Ẩn các nội dung hiện tại
        $('#result-placeholder').addClass('d-none');
        $('#result-content').addClass('d-none');
        $('#frames-section').addClass('d-none');
        $('#error-message').addClass('d-none');
        
        // Hiển thị loading
        $('#result-loading').removeClass('d-none');
        
        // Vô hiệu hóa nút submit
        $('#submit-btn').prop('disabled', true);
        
        // Tạo form data
        const formData = new FormData();
        formData.append('image1', $('#image1')[0].files[0]);
        formData.append('image2', $('#image2')[0].files[0]);
        formData.append('frames', $('#frames').val());
        
        // Gửi AJAX request
        $.ajax({
            url: '/interpolate',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Ẩn loading
                $('#result-loading').addClass('d-none');
                
                // Hiển thị kết quả
                $('#result-content').removeClass('d-none');
                
                // Cập nhật video
                $('#result-video').attr('src', response.video);
                $('#download-video').attr('href', response.video);
                
                // Hiển thị các khung hình
                displayFrames(response.frames, response.original1, response.original2);
                
                // Kích hoạt lại nút submit
                $('#submit-btn').prop('disabled', false);
            },
            error: function(xhr) {
                // Ẩn loading
                $('#result-loading').addClass('d-none');
                
                // Hiển thị lại placeholder
                $('#result-placeholder').removeClass('d-none');
                
                // Hiển thị lỗi
                const response = xhr.responseJSON || {};
                showError(response.error || 'Đã xảy ra lỗi trong quá trình xử lý. Vui lòng thử lại.');
                
                // Kích hoạt lại nút submit
                $('#submit-btn').prop('disabled', false);
            }
        });
    });
    
    // Hiển thị thông báo lỗi
    function showError(message) {
        $('#error-message').text(message).removeClass('d-none');
    }
    
    // Hiển thị các khung hình
    function displayFrames(frames, original1, original2) {
        const container = $('#frames-container');
        container.empty();
        
        // Thêm ảnh gốc đầu tiên
        const firstItem = $('<div class="frame-item">');
        firstItem.append($('<img>').attr('src', original1));
        firstItem.append($('<div class="frame-number">').text('Bắt đầu'));
        container.append(firstItem);
        
        // Thêm các khung hình trung gian
        frames.forEach((frame, index) => {
            const item = $('<div class="frame-item">');
            item.append($('<img>').attr('src', frame));
            item.append($('<div class="frame-number">').text(`${index + 1}`));
            container.append(item);
        });
        
        // Thêm ảnh gốc cuối cùng
        const lastItem = $('<div class="frame-item">');
        lastItem.append($('<img>').attr('src', original2));
        lastItem.append($('<div class="frame-number">').text('Kết thúc'));
        container.append(lastItem);
        
        // Hiển thị phần khung hình
        $('#frames-section').removeClass('d-none');
    }
});