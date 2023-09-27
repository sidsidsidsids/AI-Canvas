package com.siliconvalley.domain.google.service;

import com.siliconvalley.domain.canvas.dao.CanvasFindDao;
import com.siliconvalley.domain.canvas.domain.Canvas;
import com.siliconvalley.domain.google.dto.LabelResultDto;
import com.siliconvalley.domain.stage.domain.Score;
import com.siliconvalley.global.common.code.CommonCode;
import com.siliconvalley.global.common.dto.Response;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Transactional
@Slf4j
public class VisionDetectingService {

    private final GoogleVisionApiService visionService;
    private final CanvasFindDao canvasFindDao;

    public Score calculateCanvasScore(Long canvasId) {
        Canvas canvas = canvasFindDao.findById(canvasId);
        Map<String, Float> detectResult = visionService.detectLabels(canvas.getCanvas());
        if (detectResult.size() == 0 || canvas.getSubject().getPix2Pix().getVisionName().equals(null)) {return Score.LOW;}
        Float detectionScore = detectResult.get(canvas.getSubject().getPix2Pix().getVisionName());

        return Score.determineScore(detectionScore);
    }

    public Response testLabelDetecting(Long canvasId){
        Canvas canvas = canvasFindDao.findById(canvasId);
        List<LabelResultDto> results = new ArrayList<>();
        for (Map.Entry<String ,Float> entry : visionService.detectLabels(canvas.getCanvas()).entrySet()){
            results.add(new LabelResultDto(entry.getKey(), entry.getValue()));
        }

        return Response.of(CommonCode.GOOD_REQUEST,  results);

    }

}
