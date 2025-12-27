$(document).ready(function() {
    // Xem trước ảnh
    function previewImage(input, previewId) {
        const preview = $(`#${previewId}`);
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = $('<img>').attr('src', e.target.result);
                preview.empty().append(img);
            }
            reader.readAsDataURL(input.files[0]);
        } else {
            preview.html('<span class="text-muted">Chọn ảnh</span>');
        }
    }

    // Thêm hàm helper để format timing
    function formatTiming(timing) {
        if (!timing) return '';
    
        let html = '<div class="timing-info">';
        html += '<h4>⏱️ Thời gian xử lý:</h4>';
        html += '<table class="timing-table">';
    
        if (timing.upload_time !== undefined) {
            html += `<tr><td>Upload:</td><td>${timing.upload_time}s</td></tr>`;
        }
        if (timing.model_load_time !== undefined) {
            html += `<tr><td>Load model:</td><td>${timing.model_load_time}s</td></tr>`;
        }
        if (timing.extract_time !== undefined) {
            html += `<tr><td>Trích xuất frames:</td><td>${timing.extract_time}s</td></tr>`;
        }
        if (timing.dedup_time !== undefined) {
            html += `<tr><td>Loại bỏ trùng lặp:</td><td>${timing.dedup_time}s</td></tr>`;
        }
        if (timing.inference_time !== undefined) {
            html += `<tr><td><strong>Inference:</strong></td><td><strong>${timing.inference_time}s</strong></td></tr>`;
        }
        if (timing.postprocess_time !== undefined) {
            html += `<tr><td>Hậu xử lý:</td><td>${timing.postprocess_time}s</td></tr>`;
        }
        if (timing.total_time !== undefined) {
            html += `<tr class="total-row"><td><strong>Tổng:</strong></td><td><strong>${timing.total_time}s</strong></td></tr>`;
        }
    
        html += '</table>';
    
        // Thêm thông tin performance
        html += '<div class="performance-info">';
        if (timing.avg_time_per_pair !== undefined) {
            html += `<span class="perf-badge">⚡ ${timing.avg_time_per_pair}s/cặp</span>`;
        }
        html += '</div>';
    
        html += '</div>';
        return html;
    }
    
    // Xem trước video
    function previewVideo(input, previewId) {
        const preview = $(`#${previewId}`);
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const video = $('<video controls>').attr('src', e.target.result);
                preview.empty().append(video);
            }
            reader.readAsDataURL(input.files[0]);
        } else {
            preview.html('<span class="text-muted">Chọn video</span>');
        }
    }
    
    // Xử lý khi chọn file ảnh
    $('#image1').change(function() {
        previewImage(this, 'preview1');
    });
    
    $('#image2').change(function() {
        previewImage(this, 'preview2');
    });
    
    // Xử lý khi chọn file video
    $('#input-video').change(function() {
        previewVideo(this, 'video-preview');
    });
    
    // Xử lý khi chọn file video so sánh
    $('#compare-video1').change(function() {
        previewVideo(this, 'compare-preview1');
    });
    
    $('#compare-video2').change(function() {
        previewVideo(this, 'compare-preview2');
    });
    
    // Biến lưu trạng thái cho form nội suy ảnh
    let pendingInterpolation = null;
    
    // Xử lý form submit cho ảnh - thêm logic SSIM
    $('#upload-form').submit(function(e) {
        e.preventDefault();
        
        if (!$('#image1')[0].files[0] || !$('#image2')[0].files[0]) {
            showError('Vui lòng chọn cả hai ảnh đầu vào', 'error-message');
            return;
        }
        
        performImageInterpolation(false);
    });
    
    // Hàm thực hiện nội suy ảnh
    function performImageInterpolation(forceProcess) {
        $('#result-container').hide();
        $('#frames-card').hide();
        $('#error-message').hide();
        $('#placeholder').hide();
        $('#loading-indicator').show();
        $('#upload-form button[type="submit"]').prop('disabled', true);
        
        const formData = new FormData();
        formData.append('image1', $('#image1')[0].files[0]);
        formData.append('image2', $('#image2')[0].files[0]);
        formData.append('frames', $('#frames').val());
        formData.append('model', $('#model').val());
        formData.append('force_process', forceProcess ? 'true' : 'false');
        
        $.ajax({
            url: '/interpolate',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#loading-indicator').hide();
                
                // Kiểm tra nếu có cảnh báo SSIM
                if (response.warning) {
                    // Lưu form data để xử lý sau
                    pendingInterpolation = formData;
                    
                    // Hiển thị modal cảnh báo
                    $('#ssim-warning-message').text(response.message);
                    
                    if (response.need_resize) {
                        $('#resize-details').text(`Ảnh đã được resize từ ${response.original_sizes} về ${response.resized_to} để phù hợp với model.`);
                        $('#ssim-resize-info').show();
                    } else {
                        $('#ssim-resize-info').hide();
                    }
                    
                    const modal = new bootstrap.Modal($('#ssimWarningModal')[0]);
                    modal.show();
                    
                    $('#upload-form button[type="submit"]').prop('disabled', false);
                    $('#placeholder').show();
                } else {
                    // Xử lý thành công
                    $('#result-container').show();
                    
                    $('#result-video').attr('src', response.video);
                    $('#download-video').attr('href', response.download_video);
                    
                    displayFrames(response.frames, response.original1, response.original2);
                    
                    // Hiển thị thông tin timing
                    if (response.timing) {
                        $('#timing-info-container').html(formatTiming(response.timing)).show();
                    } else {
                        $('#timing-info-container').hide();
                    }
                    
                    // Hiển thị thông tin SSIM và resize nếu có
                    if (response.similarity !== undefined) {
                        console.log(`SSIM: ${response.similarity}, Kích thước: ${response.final_size}`);
                    }
                    
                    $('#upload-form button[type="submit"]').prop('disabled', false);
                }
            },
            error: function(xhr) {
                $('#loading-indicator').hide();
                $('#placeholder').show();
                
                const response = xhr.responseJSON || {};
                showError(response.error || 'Đã xảy ra lỗi trong quá trình xử lý', 'error-message');
                
                $('#upload-form button[type="submit"]').prop('disabled', false);
            }
        });
    }
    
    // Xử lý nút "Tiếp tục xử lý" trong modal
    $('#force-interpolate-btn').click(function() {
        const modal = bootstrap.Modal.getInstance($('#ssimWarningModal')[0]);
        modal.hide();
        performImageInterpolation(true);
    });
    
    // Xử lý form submit cho video
    $('#upload-video-form').submit(function(e) {
        e.preventDefault();
        
        if (!$('#input-video')[0].files[0]) {
            showError('Vui lòng chọn file video', 'video-error-message');
            return;
        }
        
        $('#video-result-container').hide();
        $('#samples-card').hide();
        $('#video-error-message').hide();
        $('#video-placeholder').hide();
        $('#video-loading-indicator').show();
        $('#upload-video-form button[type="submit"]').prop('disabled', true);
        
        const formData = new FormData();
        formData.append('video', $('#input-video')[0].files[0]);
        formData.append('interpolations', $('#interpolations').val());
        formData.append('fps', $('#fps').val());
        formData.append('model', $('#video-model').val());
        
        $.ajax({
            url: '/interpolate_video',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#video-loading-indicator').hide();
                $('#video-result-container').show();
                
                $('#video-result-video').attr('src', response.video);
                $('#download-result-video').attr('href', response.download_video);
                
                displayVideoStats(response.stats);
                displaySamples(response.samples);
                
                // Hiển thị thông tin timing
                if (response.timing) {
                    $('#video-timing-info-container').html(formatTiming(response.timing)).show();
                } else {
                    $('#video-timing-info-container').hide();
                }
                
                $('#upload-video-form button[type="submit"]').prop('disabled', false);
            },
            error: function(xhr) {
                $('#video-loading-indicator').hide();
                $('#video-placeholder').show();
                
                const response = xhr.responseJSON || {};
                showError(response.error || 'Đã xảy ra lỗi trong quá trình xử lý video', 'video-error-message');
                
                $('#upload-video-form button[type="submit"]').prop('disabled', false);
            }
        });
    });
    
    // Xử lý form submit cho so sánh video
    $('#comparison-form').submit(function(e) {
        e.preventDefault();
        
        if (!$('#compare-video1')[0].files[0] || !$('#compare-video2')[0].files[0]) {
            showError('Vui lòng chọn cả hai video để so sánh', 'comparison-error-message');
            return;
        }
        
        $('#comparison-result-container').hide();
        $('#comparison-error-message').hide();
        $('#comparison-placeholder').hide();
        $('#comparison-loading-indicator').show();
        $('#comparison-form button[type="submit"]').prop('disabled', true);
        
        const formData = new FormData();
        formData.append('video1', $('#compare-video1')[0].files[0]);
        formData.append('video2', $('#compare-video2')[0].files[0]);
        
        const layout = $('#layout').val();
        
        $.ajax({
            url: '/prepare_comparison',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#comparison-loading-indicator').hide();
                $('#comparison-result-container').show();
                
                displayComparisonVideos(response.video1, response.video2, layout);
                displayComparisonStats(response.stats);
                
                $('#comparison-form button[type="submit"]').prop('disabled', false);
            },
            error: function(xhr) {
                $('#comparison-loading-indicator').hide();
                $('#comparison-placeholder').show();
                
                const response = xhr.responseJSON || {};
                showError(response.error || 'Đã xảy ra lỗi trong quá trình tải video', 'comparison-error-message');
                
                $('#comparison-form button[type="submit"]').prop('disabled', false);
            }
        });
    });
    
    // Theo dõi thay đổi layout
    $('#layout').change(function() {
        const video1Src = $('#comparison-video1').attr('src');
        const video2Src = $('#comparison-video2').attr('src');
        
        if (video1Src && video2Src) {
            displayComparisonVideos(video1Src, video2Src, $(this).val());
        }
    });
    
    // Hiển thị thông báo lỗi
    function showError(message, elementId) {
        $(`#${elementId}`).text(message).show();
    }
    
    // Hiển thị các khung hình
    function displayFrames(frames, original1, original2) {
        const container = $('#frames-container');
        container.empty();
        
        const firstItem = $('<div class="frame-item">');
        firstItem.append($('<img>').attr('src', original1));
        container.append(firstItem);
        
        frames.forEach((frame) => {
            const item = $('<div class="frame-item">');
            item.append($('<img>').attr('src', frame));
            container.append(item);
        });
        
        const lastItem = $('<div class="frame-item">');
        lastItem.append($('<img>').attr('src', original2));
        container.append(lastItem);
        
        $('#frames-card').show();
    }
    
    // Hiển thị thống kê video
    function displayVideoStats(stats) {
        const container = $('#video-stats');
        container.empty();
        
        const statItems = [
            { label: 'Frames gốc', value: stats.original_frames },
            { label: 'Frames duy nhất', value: stats.unique_frames },
            { label: 'Frames cuối', value: stats.final_frames },
            { label: 'FPS', value: stats.fps }
        ];
        
        statItems.forEach(item => {
            const statDiv = $('<div class="stat-item">');
            statDiv.append($('<div class="stat-value">').text(item.value));
            statDiv.append($('<div class="stat-label">').text(item.label));
            container.append(statDiv);
        });
    }
    
    // Hiển thị các khung hình mẫu
    function displaySamples(samples) {
        const container = $('#samples-container');
        container.empty();
        
        samples.forEach((sample) => {
            const item = $('<div class="frame-item">');
            item.append($('<img>').attr('src', sample));
            container.append(item);
        });
        
        $('#samples-card').show();
    }
    
    // Hiển thị thống kê so sánh video
    function displayComparisonStats(stats) {
        const container = $('#comparison-stats');
        container.empty();
        
        const statItems = [
            { label: 'Video 1 - FPS', value: stats.video1.fps },
            { label: 'Video 1 - Frames', value: stats.video1.frames },
            { label: 'Video 1 - Độ phân giải', value: stats.video1.resolution },
            { label: 'Video 2 - FPS', value: stats.video2.fps },
            { label: 'Video 2 - Frames', value: stats.video2.frames },
            { label: 'Video 2 - Độ phân giải', value: stats.video2.resolution }
        ];
        
        statItems.forEach(item => {
            const statDiv = $('<div class="stat-item">');
            statDiv.append($('<div class="stat-value">').text(item.value));
            statDiv.append($('<div class="stat-label">').text(item.label));
            container.append(statDiv);
        });
    }
    
    // Hiển thị hai video so sánh
    function displayComparisonVideos(video1Src, video2Src, layout) {
        const wrapper = $('#comparison-videos-wrapper');
        wrapper.empty();
        
        if (layout === 'side-by-side') {
            wrapper.html(`
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="text-center mb-2">Video 1 (Gốc)</h6>
                        <div class="ratio ratio-16x9">
                            <video id="comparison-video1" controls loop preload="metadata">
                                <source src="${video1Src}" type="video/mp4">
                            </video>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-center mb-2">Video 2 (Đã xử lý)</h6>
                        <div class="ratio ratio-16x9">
                            <video id="comparison-video2" controls loop preload="metadata">
                                <source src="${video2Src}" type="video/mp4">
                            </video>
                        </div>
                    </div>
                </div>
            `);
        } else {
            wrapper.html(`
                <div>
                    <h6 class="text-center mb-2">Video 1 (Gốc)</h6>
                    <div class="ratio ratio-16x9 mb-3">
                        <video id="comparison-video1" controls loop preload="metadata">
                            <source src="${video1Src}" type="video/mp4">
                        </video>
                    </div>
                    <h6 class="text-center mb-2">Video 2 (Đã xử lý)</h6>
                    <div class="ratio ratio-16x9">
                        <video id="comparison-video2" controls loop preload="metadata">
                            <source src="${video2Src}" type="video/mp4">
                        </video>
                    </div>
                </div>
            `);
        }
        
        // Đợi cả hai video load xong metadata trước khi đồng bộ
        const video1 = $('#comparison-video1')[0];
        const video2 = $('#comparison-video2')[0];
        
        let video1Ready = false;
        let video2Ready = false;
        
        function checkBothReady() {
            if (video1Ready && video2Ready) {
                syncVideos();
            }
        }
        
        video1.addEventListener('loadedmetadata', function() {
            video1Ready = true;
            checkBothReady();
        });
        
        video2.addEventListener('loadedmetadata', function() {
            video2Ready = true;
            checkBothReady();
        });
    }
    
    // Đồng bộ hai video khi play/pause/seek
    function syncVideos() {
        const video1 = $('#comparison-video1')[0];
        const video2 = $('#comparison-video2')[0];
        
        if (!video1 || !video2) return;
        
        // Biến để tránh infinite loop
        let isSyncing = false;
        
        // Đồng bộ play
        video1.addEventListener('play', function() {
            if (!isSyncing) {
                isSyncing = true;
                video2.currentTime = video1.currentTime;
                video2.play().catch(e => console.log('Video 2 play error:', e));
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        video2.addEventListener('play', function() {
            if (!isSyncing) {
                isSyncing = true;
                video1.currentTime = video2.currentTime;
                video1.play().catch(e => console.log('Video 1 play error:', e));
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        // Đồng bộ pause
        video1.addEventListener('pause', function() {
            if (!isSyncing) {
                isSyncing = true;
                video2.pause();
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        video2.addEventListener('pause', function() {
            if (!isSyncing) {
                isSyncing = true;
                video1.pause();
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        // Đồng bộ seeking
        video1.addEventListener('seeking', function() {
            if (!isSyncing) {
                isSyncing = true;
                video2.currentTime = video1.currentTime;
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        video2.addEventListener('seeking', function() {
            if (!isSyncing) {
                isSyncing = true;
                video1.currentTime = video2.currentTime;
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        // Đồng bộ playback rate
        video1.addEventListener('ratechange', function() {
            if (!isSyncing) {
                isSyncing = true;
                video2.playbackRate = video1.playbackRate;
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        video2.addEventListener('ratechange', function() {
            if (!isSyncing) {
                isSyncing = true;
                video1.playbackRate = video2.playbackRate;
                setTimeout(() => isSyncing = false, 100);
            }
        });
        
        // Đồng bộ liên tục trong khi play để giảm lag
        let syncInterval = null;
        
        function startContinuousSync() {
            if (syncInterval) clearInterval(syncInterval);
            syncInterval = setInterval(function() {
                if (!video1.paused && !video2.paused) {
                    const timeDiff = Math.abs(video1.currentTime - video2.currentTime);
                    // Nếu chênh lệch > 0.1s thì đồng bộ lại
                    if (timeDiff > 0.1) {
                        video2.currentTime = video1.currentTime;
                    }
                }
            }, 100);
        }
        
        function stopContinuousSync() {
            if (syncInterval) {
                clearInterval(syncInterval);
                syncInterval = null;
            }
        }
        
        video1.addEventListener('play', startContinuousSync);
        video1.addEventListener('pause', stopContinuousSync);
        video1.addEventListener('ended', stopContinuousSync);
    }
    
    // Load gallery items từ localStorage khi trang được load
    loadGalleryFromStorage();
    
    // Xử lý form thêm vào trưng bày
    $('#gallery-add-form').submit(function(e) {
        e.preventDefault();
        
        if (!$('#gallery-video-original')[0].files[0] || !$('#gallery-video-bimvfi')[0].files[0]) {
            showError('Vui lòng chọn ít nhất video gốc và video BiM-VFI', 'gallery-error-message');
            return;
        }
        
        $('#gallery-error-message').hide();
        $('#gallery-loading-indicator').show();
        $('#gallery-add-form button[type="submit"]').prop('disabled', true);
        
        const formData = new FormData();
        formData.append('original', $('#gallery-video-original')[0].files[0]);
        formData.append('bimvfi', $('#gallery-video-bimvfi')[0].files[0]);
        if ($('#gallery-video-other')[0].files[0]) {
            formData.append('other', $('#gallery-video-other')[0].files[0]);
        }
        formData.append('title', $('#gallery-title').val());
        formData.append('layout', $('#gallery-layout').val());
        
        $.ajax({
            url: '/gallery_add',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#gallery-loading-indicator').hide();
                
                addGalleryItem(response);
                saveGalleryToStorage(response);
                
                // Reset form
                $('#gallery-add-form')[0].reset();
                $('#gallery-placeholder').hide();
                
                $('#gallery-add-form button[type="submit"]').prop('disabled', false);
            },
            error: function(xhr) {
                $('#gallery-loading-indicator').hide();
                
                const response = xhr.responseJSON || {};
                showError(response.error || 'Đã xảy ra lỗi khi thêm vào trưng bày', 'gallery-error-message');
                
                $('#gallery-add-form button[type="submit"]').prop('disabled', false);
            }
        });
    });
    
    // Thêm một item vào gallery
    function addGalleryItem(data) {
        const container = $('#gallery-container');
        
        const itemHtml = createGalleryItemHtml(data);
        container.append(itemHtml);
        
        // Setup video sync cho item mới
        setupGalleryVideoSync(data.id);
    }
    
    // Tạo HTML cho gallery item
    function createGalleryItemHtml(data) {
        const hasOther = data.videos.other !== null;
        const layoutClass = data.layout === 'row' ? 'gallery-videos-row' : 'gallery-videos-column';
        
        let videosHtml = `
            <div class="gallery-video-wrapper">
                <div class="gallery-video-label">Gốc</div>
                <div class="ratio ratio-16x9">
                    <video class="gallery-video" data-group="${data.id}" data-index="0" controls loop preload="metadata">
                        <source src="${data.videos.original}" type="video/mp4">
                    </video>
                </div>
            </div>
            <div class="gallery-video-wrapper">
                <div class="gallery-video-label">BiM-VFI pre-trained</div>
                <div class="ratio ratio-16x9">
                    <video class="gallery-video" data-group="${data.id}" data-index="1" controls loop preload="metadata">
                        <source src="${data.videos.bimvfi}" type="video/mp4">
                    </video>
                </div>
            </div>
        `;
        
        if (hasOther) {
            videosHtml += `
                <div class="gallery-video-wrapper">
                    <div class="gallery-video-label">BiM-VFI reproduce</div>
                    <div class="ratio ratio-16x9">
                        <video class="gallery-video" data-group="${data.id}" data-index="2" controls loop preload="metadata">
                            <source src="${data.videos.other}" type="video/mp4">
                        </video>
                    </div>
                </div>
            `;
        }
        
        return `
            <div class="gallery-item" id="gallery-item-${data.id}">
                <div class="gallery-item-title">${escapeHtml(data.title)}</div>
                <div class="gallery-item-controls">
                    <button class="btn btn-sm btn-primary gallery-play-all" data-group="${data.id}">Phát tất cả</button>
                    <button class="btn btn-sm btn-secondary gallery-pause-all" data-group="${data.id}">Tạm dừng tất cả</button>
                    <button class="btn btn-sm btn-danger gallery-remove" data-id="${data.id}">Xóa</button>
                </div>
                <div class="${layoutClass}">
                    ${videosHtml}
                </div>
            </div>
        `;
    }
    
    // Setup đồng bộ video cho một gallery item
    function setupGalleryVideoSync(groupId) {
        const videos = $(`.gallery-video[data-group="${groupId}"]`);
        
        if (videos.length === 0) return;
        
        let isSyncing = false;
        
        videos.each(function() {
            const video = this;
            
            // Play sync
            video.addEventListener('play', function() {
                if (!isSyncing) {
                    isSyncing = true;
                    const currentTime = video.currentTime;
                    videos.each(function() {
                        if (this !== video) {
                            this.currentTime = currentTime;
                            this.play().catch(e => console.log('Play error:', e));
                        }
                    });
                    setTimeout(() => isSyncing = false, 100);
                }
            });
            
            // Pause sync
            video.addEventListener('pause', function() {
                if (!isSyncing) {
                    isSyncing = true;
                    videos.each(function() {
                        if (this !== video) {
                            this.pause();
                        }
                    });
                    setTimeout(() => isSyncing = false, 100);
                }
            });
            
            // Seek sync
            video.addEventListener('seeking', function() {
                if (!isSyncing) {
                    isSyncing = true;
                    const currentTime = video.currentTime;
                    videos.each(function() {
                        if (this !== video) {
                            this.currentTime = currentTime;
                        }
                    });
                    setTimeout(() => isSyncing = false, 100);
                }
            });
        });
    }
    
    // Play all videos in a group
    $(document).on('click', '.gallery-play-all', function() {
        const groupId = $(this).data('group');
        $(`.gallery-video[data-group="${groupId}"]`).each(function() {
            this.play().catch(e => console.log('Play error:', e));
        });
    });
    
    // Pause all videos in a group
    $(document).on('click', '.gallery-pause-all', function() {
        const groupId = $(this).data('group');
        $(`.gallery-video[data-group="${groupId}"]`).each(function() {
            this.pause();
        });
    });
    
    // Remove gallery item
    $(document).on('click', '.gallery-remove', function() {
        const itemId = $(this).data('id');
        if (confirm('Bạn có chắc muốn xóa bộ video này?')) {
            $(`#gallery-item-${itemId}`).remove();
            removeGalleryFromStorage(itemId);
            
            if ($('#gallery-container .gallery-item').length === 0) {
                $('#gallery-placeholder').show();
            }
        }
    });
    
    // Lưu gallery item vào localStorage
    function saveGalleryToStorage(data) {
        let galleryItems = JSON.parse(localStorage.getItem('galleryItems') || '[]');
        galleryItems.push(data);
        localStorage.setItem('galleryItems', JSON.stringify(galleryItems));
    }
    
    // Xóa gallery item khỏi localStorage
    function removeGalleryFromStorage(itemId) {
        let galleryItems = JSON.parse(localStorage.getItem('galleryItems') || '[]');
        galleryItems = galleryItems.filter(item => item.id !== itemId);
        localStorage.setItem('galleryItems', JSON.stringify(galleryItems));
    }
    
    // Load gallery items từ localStorage
    function loadGalleryFromStorage() {
        const galleryItems = JSON.parse(localStorage.getItem('galleryItems') || '[]');
        
        if (galleryItems.length > 0) {
            $('#gallery-placeholder').hide();
            
            galleryItems.forEach(item => {
                addGalleryItem(item);
            });
        }
    }
    
    // Helper function to escape HTML
    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
    
    // Xử lý trích xuất frames từ video
    $('#extract-frames-form').submit(function(e) {
        e.preventDefault();
        
        if (!$('#extract-video')[0].files[0]) {
            showError('Vui lòng chọn video', 'extract-error');
            return;
        }
        
        $('#extract-result').hide();
        $('#extract-error').hide();
        $('#extract-loading').show();
        $('#extract-frames-form button').prop('disabled', true);
        
        const formData = new FormData();
        formData.append('video', $('#extract-video')[0].files[0]);
        
        $.ajax({
            url: '/extract_frames_from_video',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#extract-loading').hide();
                $('#extract-result').show();
                
                $('#extract-info').html(`
                    <div>Đã trích xuất <strong>${response.total_frames}</strong> frames</div>
                    <div>FPS: ${response.fps}</div>
                    <div>Frames được lưu tại: <code>${response.frames_folder}</code></div>
                    <div class="mt-2"><small>Bạn có thể chọn frames từ folder này ở Chức năng 2 bên dưới.</small></div>
                `);
                
                $('#extract-frames-form button').prop('disabled', false);
            },
            error: function(xhr) {
                $('#extract-loading').hide();
                const response = xhr.responseJSON || {};
                showError(response.error || 'Đã xảy ra lỗi khi trích xuất frames', 'extract-error');
                $('#extract-frames-form button').prop('disabled', false);
            }
        });
    });
    
    // Xử lý nội suy chuỗi frames
    $('#sequence-interpolate-form').submit(function(e) {
        e.preventDefault();
        
        const files = $('#sequence-frames')[0].files;
        if (!files || files.length < 2) {
            showError('Vui lòng chọn ít nhất 2 frames', 'sequence-error');
            return;
        }
        
        $('#sequence-result-container').hide();
        $('#sequence-samples-card').hide();
        $('#sequence-error').hide();
        $('#sequence-placeholder').hide();
        $('#sequence-loading').show();
        $('#sequence-interpolate-form button').prop('disabled', true);
        
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('frames', files[i]);
        }
        formData.append('interpolations', $('#sequence-interpolations').val());
        formData.append('fps', $('#sequence-fps').val());
        formData.append('model', $('#sequence-model').val());
        
        $.ajax({
            url: '/interpolate_frame_sequence',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#sequence-loading').hide();
                $('#sequence-result-container').show();
                
                $('#sequence-result-video').attr('src', response.video);
                $('#download-sequence-video').attr('href', response.download_video);
                
                // Hiển thị thống kê
                const statsContainer = $('#sequence-stats');
                statsContainer.empty();
                
                const statItems = [
                    { label: 'Frames đầu vào', value: response.stats.input_frames },
                    { label: 'Frames đầu ra', value: response.stats.output_frames },
                    { label: 'FPS', value: response.stats.fps }
                ];
                
                statItems.forEach(item => {
                    const statDiv = $('<div class="stat-item">');
                    statDiv.append($('<div class="stat-value">').text(item.value));
                    statDiv.append($('<div class="stat-label">').text(item.label));
                    statsContainer.append(statDiv);
                });
                
                // Hiển thị thông tin timing
                if (response.timing) {
                    $('#sequence-timing-info-container').html(formatTiming(response.timing)).show();
                } else {
                    $('#sequence-timing-info-container').hide();
                }
                
                // Hiển thị samples
                if (response.samples && response.samples.length > 0) {
                    const samplesContainer = $('#sequence-samples-container');
                    samplesContainer.empty();
                    
                    response.samples.forEach(sample => {
                        const img = $('<img>').attr('src', sample).css({
                            'height': '100px',
                            'border-radius': '4px',
                            'border': '1px solid #dee2e6'
                        });
                        samplesContainer.append(img);
                    });
                    
                    $('#sequence-samples-card').show();
                }
                
                $('#sequence-interpolate-form button').prop('disabled', false);
            },
            error: function(xhr) {
                $('#sequence-loading').hide();
                $('#sequence-placeholder').show();
                const response = xhr.responseJSON || {};
                showError(response.error || 'Đã xảy ra lỗi khi nội suy chuỗi frames', 'sequence-error');
                $('#sequence-interpolate-form button').prop('disabled', false);
            }
        });
    });
});