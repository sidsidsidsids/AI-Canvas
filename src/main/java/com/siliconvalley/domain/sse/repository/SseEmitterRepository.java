package com.siliconvalley.domain.sse.repository;

import org.springframework.stereotype.Repository;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Repository
public class SseEmitterRepository {

    private final Map<String, SseEmitter> sseEmitterMap = new ConcurrentHashMap<>();

    public void save(String id, SseEmitter sseEmitter) {
        sseEmitterMap.put(id, sseEmitter);
    }

    public void delete(String id) {
        sseEmitterMap.remove(id);
    }

    public SseEmitter findById(String id) {
        return sseEmitterMap.get(id);
    }
}
