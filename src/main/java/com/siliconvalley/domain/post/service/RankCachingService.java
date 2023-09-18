package com.siliconvalley.domain.post.service;

import com.siliconvalley.domain.item.subject.dao.SubjectFindDao;
import com.siliconvalley.domain.item.subject.domain.Subject;
import com.siliconvalley.domain.post.code.RankingCode;
import com.siliconvalley.domain.post.dto.RankingCachingDto;
import com.siliconvalley.domain.post.dto.RankingPeriodDto;
import com.siliconvalley.global.common.dto.Response;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.concurrent.TimeUnit;

@Service
@Transactional
@RequiredArgsConstructor
@Slf4j
public class RankCachingService {

    private final RedisTemplate<String, RankingCachingDto> redisTemplate;
    private final SubjectFindDao subjectFindDao;

    public Response getRankingThisWeekBySubject(Long subjectId){
        Subject subject = subjectFindDao.findById(subjectId);
        RankingCachingDto rankingCachingDto = redisTemplate.opsForList().index(generateRedisKey(subject.getSubjectName()), -1);
        return Response.of(RankingCode.GET_RANKING_SUCCESS, rankingCachingDto);
    }

    public void cachingRankToRedis(RankingCachingDto rankingCachingDto){
        redisTemplate.opsForList().rightPush(generateRedisKey(rankingCachingDto.getSubjectName()), rankingCachingDto);
        // 키의 만료 시간을 4주로 설정
        redisTemplate.expire(generateRedisKey(rankingCachingDto.getSubjectName()), 28, TimeUnit.DAYS);
    }

    public String getTopPostThisWeek(String subjectName){
        RankingCachingDto rankingCachingDto = redisTemplate.opsForList().index(generateRedisKey(subjectName), -1);
        log.info(rankingCachingDto.getRankerList().size() + "개의 리스트");
        log.info(rankingCachingDto.getRankerList().get(0).getCanvasUrl());
        log.info(rankingCachingDto.getRankerList().get(0).getPostId() + "번 게시물");
        return rankingCachingDto.getRankerList().get(0).getCanvasUrl();
    }

    private String generateRedisKey(String subjectName) {
        RankingPeriodDto dateDto = RankingPeriodDto.builder().build();
        return "week:" + dateDto.getWeekOfYear() + ":day:" + dateDto.getDayOfWeek() + ':' + subjectName + ":rank";
    }
}
