package com.siliconvalley.global.config.redis;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.siliconvalley.domain.post.dto.RankingCachingDto;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.RedisStandaloneConfiguration;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.Jackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;


@Configuration
public class RedisCachingConfig {

    @Value("${spring.second-redis.cache.host}")
    private String hostName;

    @Value("${spring.second-redis.cache.port}")
    private int port;

    @Bean
    public RedisConnectionFactory redisCachingConnectionFactory() {
        RedisStandaloneConfiguration redisStandaloneConfiguration
                = new RedisStandaloneConfiguration();
        redisStandaloneConfiguration.setHostName(hostName);
        redisStandaloneConfiguration.setPort(port);
        return new LettuceConnectionFactory(redisStandaloneConfiguration);
    }

    @Bean
    public RedisTemplate<String, RankingCachingDto> rankingCachingRedisTemplate(RedisConnectionFactory redisConnectionFactory){
        RedisTemplate<String, RankingCachingDto> template =new RedisTemplate<>();
        template.setConnectionFactory(redisConnectionFactory);

        Jackson2JsonRedisSerializer<RankingCachingDto> serializer =new Jackson2JsonRedisSerializer<>(RankingCachingDto.class);
        serializer.setObjectMapper(new ObjectMapper());

        template.setValueSerializer(serializer);
        template.setKeySerializer(new StringRedisSerializer());
        return template;
    }

}